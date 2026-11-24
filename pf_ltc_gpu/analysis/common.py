# -*- coding: utf-8 -*-
"""Shared helpers of the analysis tools: run context, q-point lists, phonons at arbitrary q, HDF5 output.

Analysis only: the kernels / GPUConductivity are imported read-only; nothing here is used by the kappa path.

Conventions (same as the kappa path / phono3py)
  q-points   reduced coordinates of the primitive cell of the calculation; Cartesian q = q_red @ inv(cell).T in 1/A
             WITHOUT the factor 2 pi (phonopy convention)
  frequency  THz (ordinary frequency, not angular); imaginary modes are negative
  eigenvector columns of the dynamical matrix D(q) (phonopy phase convention exp(2 pi i q.(r_j - r_i)))
  velocity   THz * Angstrom (x 100 = m/s)
"""
import os
import time
from pathlib import Path

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import build_setup, load_config, log


class Context:
    """Setup + device objects of one analysis run.

    c   : GPUConductivity on the mesh of the config (None for grid-free tools)
    dm  : dynamical matrix (with NAC when the case has it), dm0 : without NAC (original fc2) or None
    """

    def __init__(self, conf, s, device, precision, mem_budget, c=None):
        self.conf, self.s, self.device, self.precision = conf, s, torch.device(device), precision
        self.mem_budget = mem_budget
        self.c = c
        self.cdtype = torch.complex64 if precision == "mixed" else torch.complex128
        if c is not None:
            self.dm, self.dm0, self._fn = c.dm, c.dm0, c.fn
        else:
            dev = self.device
            nac = s.extra.get("nac")
            if nac is not None:
                self.dm = K.DynMat(nac["fc_short_range"], s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev, nac=nac)
                self.dm0 = K.DynMat(s.fc2, s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev)
            else:
                self.dm = K.DynMat(s.fc2, s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev)
                self.dm0 = None
            self._fn = None
        self.n_p = len(s.masses)
        self.nb = 3 * self.n_p
        self.temps = torch.as_tensor(s.temperatures, device=self.device, dtype=torch.float64)
        self.lattice = torch.as_tensor(s.lattice, device=self.device, dtype=torch.float64)  # rows = a, b, c
        self.rec = torch.linalg.inv(self.lattice)  # columns = reciprocal vectors (no 2 pi)

    @property
    def fn(self):
        """Fc3Normal (built on first use for grid-free tools)."""
        if self._fn is None:
            s = self.s
            self._fn = K.Fc3Normal(s.fc3, s.svecs, s.multi, s.masses, s.p2s, s.s2pp, s.all_shortest,
                                   s.make_r0_average, s.cutoff_frequency, self.device, cdtype=self.cdtype,
                                   x_budget=self.mem_budget / 4, band_block=self.conf.get("chunk_band"))
        return self._fn

    def to_cart(self, q):
        """reduced -> Cartesian (1/A, no 2 pi)."""
        return q @ self.rec.T

    def to_reduced(self, qc):
        return qc @ self.lattice.T

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def peak_gb(self):
        return torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0


def add_common_args(a, outdir):
    a.add_argument("config", nargs="?", default=None)
    a.add_argument("--config", dest="config_opt", default=None, help="ltc_config.py of the case (default ./ltc_config.py)")
    a.add_argument("--device", default=None, help="cuda:0, cpu, ... (default: config, else cuda:0)")
    a.add_argument("--precision", default=None, choices=["fp64", "mixed"])
    a.add_argument("--mem-gb", type=float, default=None, help="memory budget (default 60%% of the free GPU memory)")
    a.add_argument("--threads", type=int, default=None)
    a.add_argument("--outdir", default=outdir)
    a.add_argument("--temperatures", default=None, metavar="T1,T2,..", help="override the config temperatures (K)")
    a.add_argument("--rotational-sum-rule", default=None, metavar="MODE",
                   help="optional (default off): impose the rotational sum rules on the force constants first; "
                        "MODE = born | born-huang | born-huang+fc3 (see pf_ltc_gpu/rotsum.py)")


def add_q_args(a):
    a.add_argument("--qpoints", default=None, metavar="'q1x,q1y,q1z;q2x,...'", help="list of q-points (reduced)")
    a.add_argument("--qpoints-file", default=None, help="text file, one q-point per line (3 columns)")
    a.add_argument("--path", default=None, metavar="'q1:q2:q3'",
                   help="straight segments between the given q-points, e.g. '0,0,0:0.5,0,0.5:0.5,0.25,0.75'")
    a.add_argument("--path-npoints", type=int, default=51, help="points per path segment, end points included")
    a.add_argument("--cartesian", action="store_true",
                   help="the q-points are Cartesian, in 1/Angstrom without 2 pi (default: reduced coordinates)")


def _vec(txt):
    v = [float(eval(x, {"__builtins__": {}}, {})) for x in txt.replace("[", "").replace("]", "").split(",")]
    if len(v) != 3:
        raise ValueError(f"expected 3 numbers, got '{txt}'")
    return v


def parse_qpoints(args, lattice=None):
    """(q (n, 3) reduced, distance (n,) along the list in 1/A or None, segment start indices)."""
    q, seg = None, None
    if getattr(args, "qpoints", None):
        q = np.array([_vec(t) for t in args.qpoints.split(";") if t.strip()], dtype="double")
    elif getattr(args, "qpoints_file", None):
        q = np.loadtxt(args.qpoints_file, dtype="double", ndmin=2)[:, :3]
    elif getattr(args, "path", None):
        nodes = np.array([_vec(t) for t in args.path.split(":")], dtype="double")
        n = max(2, int(args.path_npoints))
        q = np.concatenate([nodes[i] + (nodes[i + 1] - nodes[i]) * np.linspace(0, 1, n)[:, None]
                            for i in range(len(nodes) - 1)])
        seg = np.arange(len(nodes) - 1) * n
    if q is None:
        return None, None, None
    if getattr(args, "cartesian", False):
        q = q @ np.asarray(lattice).T
    dist = None
    if lattice is not None:
        qc = q @ np.linalg.inv(np.asarray(lattice)).T
        d = np.linalg.norm(np.diff(qc, axis=0), axis=1) if len(q) > 1 else np.zeros(0)
        if seg is not None:  # no jump between segments
            for i in seg[1:]:
                d[i - 1] = 0.0
        dist = np.concatenate([[0.0], np.cumsum(d)])
    return q, dist, seg


def make_context(args, tool, mesh="config", need_conductivity=True):
    """Build the setup (CPU) and the device objects.

    mesh="config" or [n1, n2, n3]: mesh of the config / given mesh, with GPUConductivity (phonons on the whole
    mesh in memory) when need_conductivity; mesh=None: grid-free (a 1x1x1 grid is given to build_setup, nothing of it is used)."""
    nthreads = args.threads or int(os.environ.get("PF_GPU_THREADS", "1"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(nthreads)
    torch.set_num_threads(nthreads)
    from ..cli import make_fc_builder
    from ..conductivity import GPUConductivity

    conf = load_config(args.config)
    if getattr(args, "temperatures", None):
        conf["temperatures"] = [float(t) for t in args.temperatures.split(",")]
    if getattr(args, "rotational_sum_rule", None):
        conf["rotational_sum_rule"] = args.rotational_sum_rule
    device = args.device or conf.get("device", "cuda:0")
    precision = args.precision or conf.get("precision", "fp64")
    log(f"=== PF-GPU analyze {tool} (PID {os.getpid()}) device={device} precision={precision} threads={nthreads}")
    if device.startswith("cuda"):
        log(f"[GPU] {torch.cuda.get_device_name(torch.device(device))}")
    t0 = time.time()
    s = build_setup(conf, mesh_override=None if mesh == "config" else (mesh or [1, 1, 1]),
                    fc_builder=make_fc_builder(conf, device))
    mem = args.mem_gb
    if mem is None:
        mem = 0.6 * torch.cuda.mem_get_info(torch.device(device))[0] / 1e9 if device.startswith("cuda") else 4.0
    c = None
    if mesh is not None and need_conductivity:
        c = GPUConductivity(s, device=device, precision=precision, mem_budget_gb=mem,
                            chunk_qp=conf.get("chunk_qp"), chunk_band=conf.get("chunk_band"), sigma=conf.get("sigma"),
                            average_degenerate_weights=bool(conf.get("average_degenerate_weights", False)),
                            thm_vertex_limit=bool(conf.get("thm_vertex_limit", False)))
    ctx = Context(conf, s, device, precision, mem * 1e9, c)
    ctx.t_setup = time.time() - t0
    log(f"[Setup] n_atom={ctx.n_p} nb={ctx.nb} NAC={ctx.dm.nac is not None} mem_budget={mem:.1f} GB "
        f"({ctx.t_setup:.1f} s)")
    return ctx


# ---------------------------------------------------------------------------
# Phonons at arbitrary q
# ---------------------------------------------------------------------------
def wrap_bz(q, rec):
    """Shift each reduced q by a reciprocal lattice vector so that |q + G| (Cartesian) is the smallest among the
    27 nearest images (first Brillouin zone for ordinary cells)."""
    r = q - torch.round(q)
    sh = torch.tensor([[i, j, k] for i in (0, -1, 1) for j in (0, -1, 1) for k in (0, -1, 1)],
                      device=q.device, dtype=q.dtype)
    cand = r[:, None, :] + sh[None]
    n = torch.linalg.norm(cand @ rec.T, dim=-1)
    best = n.argmin(dim=1)
    return cand[torch.arange(len(q), device=q.device), best]


def dynmat_at(dm, q):
    """D(q) (B, nb, nb) at arbitrary reduced q.

    Without NAC this is dm(q). With NAC, kernels.DynMat drops the non-analytic term when |q + G| < 1e-5 1/A;
    here a q closer than that to a reciprocal lattice point (but not on it) gets the direction-dependent limit
    along q + G instead, so that small-|q| scans are continuous. Exactly at a lattice point no non-analytic term
    is added (same as the mesh code and phono3py without a q-direction)."""
    if dm.nac is None:
        return dm(q)
    r = q - torch.round(q)
    n2 = ((r @ dm.nac["rec"].T) ** 2).sum(-1)
    exact = n2 < 1e-24
    near = (n2 < 1e-10) & ~exact
    if not bool(near.any()):
        return dm(q)
    unit = torch.zeros_like(r)
    unit[:, 0] = 1.0
    D = dm(q, q_dir=torch.where(exact[:, None], unit, r))
    if bool(exact.any()):
        D[exact] = dm(q[exact])
    return D


def phonons_at(ctx, q, eigvecs=True, chunk=None, dm=None):
    """Frequencies (THz) (B, nb) [and eigenvectors (B, nb, nb), columns] at reduced q (B, 3) on the device."""
    dm = dm or ctx.dm
    q = torch.as_tensor(q, device=ctx.device, dtype=torch.float64).reshape(-1, 3)
    if chunk is None:
        chunk = max(1, int((ctx.mem_budget / 2) // (dm.n_p * dm.n_s * 9 * 16 * 2 * (8 if dm.nac is not None else 1))))
        chunk = min(chunk, 16384)
    fs, es = [], []
    for i in range(0, len(q), chunk):
        D = dynmat_at(dm, q[i:i + chunk])
        # eigh also when only frequencies are wanted: same solver as the mesh code (identical frequencies), and the
        # batched eigvalsh of cusolver asked for > 100 GB of workspace for 1e5 6x6 matrices (2026-11-24)
        w, v = torch.linalg.eigh(D, UPLO="L")
        if eigvecs:
            es.append(v)
        del v
        fs.append(torch.sqrt(torch.abs(w)) * torch.sign(w) * ctx.s.factor)
    f = torch.cat(fs)
    return (f, torch.cat(es)) if eigvecs else f


def mode_heat_capacity(f, temps, cutoff):
    """Mode heat capacity (nT, ..., nb) in eV/K (same rule as GPUConductivity.heat_capacity)."""
    from phonopy import units as U
    t = temps.reshape((-1,) + (1,) * f.dim())
    fe = f[None] * U.THzToEv
    cond = (fe < 100 * t * U.Kb) & (fe > cutoff * U.THzToEv)
    tt = torch.where(cond, t.expand_as(cond), torch.full_like(fe, 10000.0))
    x = fe / (U.Kb * tt)
    ex = torch.exp(x)
    cv = U.Kb * x ** 2 * ex / (ex - 1.0) ** 2
    return torch.where(cond, cv, torch.zeros_like(cv))


def degenerate_mask(f, tol=1e-4):
    """(..., nb, nb) bool: bands of the same degenerate set (consecutive bands closer than tol THz, as phonopy)."""
    ids = K.degenerate_ids(f, cutoff=tol)
    return ids[..., :, None] == ids[..., None, :]


def degenerate_mean(vals, f, tol=1e-4):
    """Average vals (..., nb) [band axis last, leading axes broadcast against f (..., nb)] over degenerate sets."""
    A = K.degenerate_average_matrix(K.degenerate_ids(f, cutoff=tol)).to(vals.dtype)
    return torch.einsum("...ij,...j->...i", A, vals)


def structure_datasets(s):
    cell = s.ph3.primitive
    return dict(lattice=np.array(cell.cell), positions=np.array(cell.positions),
                scaled_positions=np.array(cell.scaled_positions), symbols=np.array(list(cell.symbols), dtype="S3"),
                masses=np.array(cell.masses), volume=float(cell.volume))


def write_hdf5(fn, data, attrs=None, units=None):
    """Write a dict of arrays; units: {dataset: unit string} stored as the attribute 'units' of the dataset."""
    import h5py
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    tmp = fn + ".tmp"
    with h5py.File(tmp, "w") as h:
        for k, v in data.items():
            v = np.asarray(v)
            if v.dtype.kind == "U":
                v = v.astype("S")
            d = h.create_dataset(k, data=v, compression="gzip" if v.size > 1000 else None)
            if units and k in units:
                d.attrs["units"] = units[k]
        for k, v in (attrs or {}).items():
            h.attrs[k] = v
    os.replace(tmp, fn)
    log(f"[Save] {fn}")


def resolve_config(args):
    args.config = args.config_opt or args.config or "ltc_config.py"
    if not Path(args.config).exists():
        raise SystemExit(f"[Error] {args.config} not found.")
    return args
