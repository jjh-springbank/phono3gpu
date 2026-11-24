# -*- coding: utf-8 -*-
"""Three-phonon linewidth of single modes at arbitrary (off-mesh) q-points, with a streamed partner mesh.

    pf-run-gpu analyze linewidth --config ltc_config.py --outdir out_lw
        --qpoints '0.01,0,0; 0.013,0.02,0' | --qpoints-file F | --path ..     q-points of the modes (any q)
        [--bands 0,1,2]            bands at each q (default all; degenerate partners are added)
        [--mesh n1,n2,n3]          mesh of the partner q-points (default: mesh of the config); any density
        [--method tetrahedron|gaussian|lorentzian] [--sigma S]   energy conservation; S in THz = standard deviation
                                   of the Gaussian or half width at half maximum of the Lorentzian
        [--mesh-check N]           compare with the mesh linewidths of the kappa path at N irreducible mesh points
        [--chunk B]                partner q-points per batch (default from the memory budget)

    pf-run-gpu analyze attenuation --config ltc_config.py --outdir out_att
        [--directions '1,0,0; 1,1,0; 1,1,1'] (Cartesian)  [--qmin 0.002 --qmax 0.05 --nq 8 | --qnorm q1,q2,..]
        [--mesh ..] [--method ..] [--sigma ..]
    linewidth and attenuation of the three acoustic modes at q = |q| n, |q| in 1/Angstrom without 2 pi.

Method.  For a mode (q0, j0) the imaginary self-energy is the sum over all partner points q1 of a regular mesh,
with q2 = -q0 - q1 (so the second partner sits on the same mesh shifted by -q0):
    gamma(q0 j0) = unit / N * sum_q1 sum_{j1 j2} |Phi(q0 j0, q1 j1, q2 j2)|^2 / (f0 f1 f2)
                   * [(n1 + n2 + 1) d(f0 - f1 - f2) + (n1 - n2) (d(f0 + f1 - f2) - d(f0 - f1 + f2))]
same expression and units as the kappa path (phono3py imag self-energy, THz, lifetime = 1 / (4 pi gamma)).
Energy conservation d(): linear tetrahedron method over the q1 mesh (vertex frequencies f1(q1 + dk),
f2(q2 - dk) computed exactly at the shifted points), Gaussian, or Lorentzian of fixed width.
Streaming: the partner points are generated batch by batch; phonons at q1, q2 and at the tetrahedron vertices
are solved on the fly, so nothing of the size of the mesh is stored on the device (or on the host), and the
mesh may be as dense as time allows. No symmetry reduction is used (all N partner points are summed).
For q0 on the mesh and the tetrahedron method the result equals the mesh linewidth of the kappa path.
Only diagonal meshes [n1, n2, n3] (no generalized regular grids).

Output linewidth.hdf5
  qpoint (nq, 3) reduced; band_index (nsel,); frequency (nq, nb) THz; group_velocity (nq, nb, 3) THz*Angstrom
  gamma (nT, nq, nsel) THz, averaged over degenerate modes; temperature (nT,) K; mesh (3,); attrs method, sigma
Output attenuation.hdf5 (bands = three lowest modes; n = direction)
  direction (nd, 3); qnorm (nk,) 1/Angstrom (no 2 pi); qpoint (nd, nk, 3) reduced
  frequency (nd, nk, 3) THz; gamma (nT, nd, nk, 3) THz; velocity_parallel (nd, nk, 3) = v_g . n, THz*Angstrom
  quality_factor (nT, nd, nk, 3) = f / (2 gamma)
  attenuation_amplitude (nT, nd, nk, 3) = 2 pi gamma / (v_g . n) in 1/m (Np/m);  attenuation_dB_per_cm = 0.0868589 * that
Note on the physics: with tetrahedron / Gaussian energy conservation this is the Landau-Rumer regime (thermal
phonons with lifetime long against the period of the wave). The Lorentzian of width S models a finite lifetime
of the thermal phonons (S ~ gamma_1 + gamma_2) by one constant; it is not a full Akhiezer treatment.
"""
import json
import time

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import log
from . import common as C
from .qpoints import velocities_at


def lorentzian(x, w):
    return (w / np.pi) / (x * x + w * w)


class LinewidthQ:
    def __init__(self, ctx, mesh, method="tetrahedron", sigma=None, chunk=None):
        from phono3py.other.tetrahedron_method import get_tetrahedra_relative_grid_address
        self.ctx = ctx
        s, dev = ctx.s, ctx.device
        self.mesh = np.array(mesh, dtype="int64")
        self.N = int(np.prod(self.mesh))
        self.method, self.sigma, self.chunk = method, sigma, chunk
        if method != "tetrahedron" and not sigma:
            raise ValueError("--sigma is needed for the gaussian / lorentzian methods")
        self.vertex_limit = bool(ctx.conf.get("thm_vertex_limit", False))
        self.avg_deg = bool(ctx.conf.get("average_degenerate_weights", False))
        self.nb = ctx.nb
        micro = np.linalg.inv(np.asarray(s.lattice)) @ np.diag(1.0 / self.mesh)  # microzone lattice (columns)
        rel = np.array(get_tetrahedra_relative_grid_address(micro), dtype="int64")  # (24, 4, 3)
        self.rel = rel
        U, inv = np.unique(rel.reshape(-1, 3), axis=0, return_inverse=True)
        self.n_u = len(U)
        self.U_addr = torch.as_tensor(U, device=dev)  # (nU, 3) integer offsets of the tetrahedron vertices
        self.u_zero = int(np.where((U == 0).all(axis=1))[0][0])
        self.u_index = torch.as_tensor(inv.reshape(24, 4), device=dev)
        self.mesh_t = torch.as_tensor(self.mesh, device=dev)
        num_grid_s = float(np.prod(s.bz_grid.D_diag))
        self.unit = s.pp_unit_conversion * num_grid_s / self.N * s.ise_unit_conversion
        self.thm_elems = max(2 ** 20, int(ctx.mem_budget / 4 // (40 * 8)))
        self.t_last = {}

    def _chunk(self, nsel):
        if self.chunk:
            return int(self.chunk)
        nb = self.nb
        csz = 8 if self.ctx.cdtype == torch.complex64 else 16
        per = 8 * nb ** 3 * csz + 10 * nsel * nb ** 2 * 8 + 2 * self.n_u * nb * nb * 16 * 3 * (1 if self.method == "tetrahedron" else 0)
        if self.method == "tetrahedron":
            per += nb ** 2 * 3 * 24 * 4 * 8 * 6 + nsel * nb ** 2 * 3 * 8 * 4
        return max(1, min(int(0.5 * self.ctx.mem_budget // per), 200000))

    def partner_points(self, i0, i1):
        """Mesh points with index i0..i1-1 and the tetrahedron vertices around them.

        Returns q1 (B, 3) reduced in [0, 1); qu (n, 3): the distinct mesh points among the centres and their
        vertices; vmap (B, nU): position in qu of vertex u of point b; cen (B,): position of the centre."""
        k = torch.arange(i0, i1, device=self.ctx.device)
        m = self.mesh_t
        a = torch.stack((k % m[0], (k // m[0]) % m[1], k // (m[0] * m[1])), dim=1)
        mf = m.to(torch.float64)[None]
        if self.method != "tetrahedron":
            return a.to(torch.float64) / mf, None, None, None
        av = (a[:, None, :] + self.U_addr[None]) % m[None, None, :]  # (B, nU, 3)
        idx = av[..., 0] + m[0] * (av[..., 1] + m[1] * av[..., 2])
        uniq, inv = torch.unique(idx.reshape(-1), return_inverse=True)
        au = torch.stack((uniq % m[0], (uniq // m[0]) % m[1], uniq // (m[0] * m[1])), dim=1)
        vmap = inv.reshape(idx.shape)
        return a.to(torch.float64) / mf, au.to(torch.float64) / mf, vmap, vmap[:, self.u_zero]

    def _tick(self, key, t0):
        self.ctx.sync()
        self.t_last[key] = self.t_last.get(key, 0.0) + time.time() - t0
        return time.time()

    def gamma(self, q0, bands=None, progress=False):
        """gamma (nT, nsel) in THz of the modes 'bands' at the reduced q-point q0 (3,). Also returns f0 (nb,)."""
        ctx = self.ctx
        s, dev, nb = ctx.s, ctx.device, self.nb
        fn = ctx.fn
        temps = ctx.temps
        nT = len(temps)
        q0 = torch.as_tensor(q0, device=dev, dtype=torch.float64).reshape(1, 3)
        f0_all, e0 = C.phonons_at(ctx, q0)
        f0_all, e0 = f0_all[0], e0[0]
        sel = torch.arange(nb, device=dev) if bands is None else torch.as_tensor(bands, device=dev)
        nsel = len(sel)
        f0 = f0_all[sel]
        E0 = (e0 * fn.inv_sqrt_m[:, None]).to(fn.cdtype)[:, sel]  # (nb, nsel)
        ise = torch.zeros((nT, nsel), device=dev, dtype=torch.float64)
        B = self._chunk(nsel)
        csz = 8 if fn.cdtype == torch.complex64 else 16
        t0 = time.time()
        for i0 in range(0, self.N, B):
            tt = time.time()
            q1, qu, vmap, cen = self.partner_points(i0, min(i0 + B, self.N))
            Bc = len(q1)
            if qu is None:
                q1w, q2w = C.wrap_bz(q1, ctx.rec), C.wrap_bz(-q0 - q1, ctx.rec)
                f1, e1 = C.phonons_at(ctx, q1w)
                f2, e2 = C.phonons_at(ctx, q2w)
            else:  # phonons once per distinct mesh point (centres and vertices), on the mesh and on the shifted mesh
                qaw, qbw = C.wrap_bz(qu, ctx.rec), C.wrap_bz(-q0 - qu, ctx.rec)
                fa, ea = C.phonons_at(ctx, qaw)
                fb, eb = C.phonons_at(ctx, qbw)
                q1w, q2w, f1, e1, f2, e2 = qaw[cen], qbw[cen], fa[cen], ea[cen], fb[cen], eb[cen]
                del ea, eb
            tt = self._tick("phonons", tt)
            qv = torch.stack((q0.expand(Bc, 3), q1w, q2w), dim=1)
            E = torch.stack((e1, e1, e2), dim=1)  # slot 0 is not used by the partner contraction
            E = (E * fn.inv_sqrt_m[None, None, :, None]).to(fn.cdtype)
            if 8 * Bc * nb ** 3 * csz > 0.8 * ctx.mem_budget:
                Y = fn.contract_partners(qv, E)
            else:
                F = fn.fc3_reciprocal(qv)
                X1 = torch.einsum("bijk,bkc->bijc", F, E[:, 2])
                del F
                Y = torch.einsum("bijc,bjm->bimc", X1, E[:, 1])
                del X1
            X = torch.einsum("bimc,il->blmc", Y, E0)  # (B, nsel, nb, nb)
            del Y, E
            ok0, ok1, ok2 = f0 > fn.cutoff, f1 > fn.cutoff, f2 > fn.cutoff
            one = torch.ones((), device=dev, dtype=torch.float64)
            fff = (torch.where(ok0, f0, one)[None, :, None, None] * torch.where(ok1, f1, one)[:, None, :, None]
                   * torch.where(ok2, f2, one)[:, None, None, :])
            okm = ok0[None, :, None, None] & ok1[:, None, :, None] & ok2[:, None, None, :]
            pp = torch.where(okm, (X.real.to(torch.float64) ** 2 + X.imag.to(torch.float64) ** 2) / fff,
                             torch.zeros_like(fff))
            del X, fff, okm
            tt = self._tick("vertex", tt)
            if self.method == "tetrahedron":
                f1v = fa[vmap][:, self.u_index].permute(0, 3, 1, 2)  # (B, nb, 24, 4): f1 at q1 + dk
                f2v = fb[vmap][:, self.u_index].permute(0, 3, 1, 2)  # f2 at -q0 - (q1 + dk) = q2 - dk
                g1, g2 = K.triplet_integration_weights(f0, f1v, f2v, max_elems=self.thm_elems,
                                                       vertex_limit=self.vertex_limit)
                del f1v, f2v, fa, fb
            else:
                a, b, c = f0[None, :, None, None], f1[:, None, :, None], f2[:, None, None, :]
                d = (lambda x: K.gaussian(x, self.sigma)) if self.method == "gaussian" else (lambda x: lorentzian(x, self.sigma))
                g1 = d(a - b - c)
                g2 = d(a + b - c) - d(a - b + c)
            if self.avg_deg:
                A1 = K.degenerate_average_matrix(K.degenerate_ids(f1))
                A2 = K.degenerate_average_matrix(K.degenerate_ids(f2))
                g1 = torch.einsum("bjJ,biJK,bkK->bijk", A1, g1, A2)
                g2 = torch.einsum("bjJ,biJK,bkK->bijk", A1, g2, A2)
            n1 = K.occupations(f1, temps, s.cutoff_frequency)
            n2 = K.occupations(f2, temps, s.cutoff_frequency)
            P1, P2 = g1 * pp, g2 * pp
            del pp, g1, g2
            S1j, S1k, S2j, S2k = P1.sum(dim=3), P1.sum(dim=2), P2.sum(dim=3), P2.sum(dim=2)
            del P1, P2
            ise += torch.einsum("tbj,bij->ti", n1 + 1, S1j) + torch.einsum("tbk,bik->ti", n2, S1k)
            ise += torch.einsum("tbj,bij->ti", n1, S2j) - torch.einsum("tbk,bik->ti", n2, S2k)
            self._tick("weights", tt)
            if progress and (i0 // B) % max(1, (self.N // B) // 5) == 0:
                log(f"[LW] partner points {min(i0 + B, self.N)}/{self.N} ({time.time() - t0:.1f} s)")
        g = ise * self.unit
        g = C.degenerate_mean(g, f0[None, :]) if self._sets_complete(f0_all, sel) else g
        return g, f0_all

    @staticmethod
    def _sets_complete(f_all, sel):
        ids = K.degenerate_ids(f_all)
        chosen = torch.zeros_like(ids, dtype=torch.bool)
        chosen[sel] = True
        return bool((chosen[ids] == chosen).all()) and bool((sel[1:] > sel[:-1]).all() if len(sel) > 1 else True)

    def complete_bands(self, q0, bands):
        """bands plus the other members of their degenerate sets at q0 (sorted)."""
        f = C.phonons_at(self.ctx, torch.as_tensor(q0, dtype=torch.float64).reshape(1, 3), eigvecs=False)[0]
        ids = K.degenerate_ids(f).cpu().numpy()
        want = set(ids[list(bands)].tolist())
        return [j for j in range(self.nb) if ids[j] in want]

    def run(self, qpoints, bands=None, progress=True):
        """gamma (nT, nq, nsel), frequency (nq, nb), bands used. With bands given, the union of the completed sets
        over all q-points is used (so every q has the same band list)."""
        if bands is not None:
            allb = set()
            for q in qpoints:
                allb.update(self.complete_bands(q, bands))
            bands = sorted(allb)
        nsel = self.nb if bands is None else len(bands)
        nT = len(self.ctx.temps)
        g = np.zeros((nT, len(qpoints), nsel))
        f = np.zeros((len(qpoints), self.nb))
        t0 = time.time()
        for i, q in enumerate(qpoints):
            gi, fi = self.gamma(q, bands)
            g[:, i], f[i] = gi.cpu().numpy(), fi.cpu().numpy()
            if progress:
                el = time.time() - t0
                log(f"[LW] q {i + 1}/{len(qpoints)} [{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}] ({el:.1f} s, "
                    f"eta {el / (i + 1) * (len(qpoints) - i - 1):.1f} s)")
        return g, f, (list(range(self.nb)) if bands is None else bands)


def _add_lw_args(a):
    a.add_argument("--mesh", default=None, metavar="n1,n2,n3", help="partner mesh (default: mesh of the config)")
    a.add_argument("--method", default="tetrahedron", choices=["tetrahedron", "gaussian", "lorentzian"])
    a.add_argument("--sigma", type=float, default=None,
                   help="THz: Gaussian standard deviation or Lorentzian half width at half maximum")
    a.add_argument("--chunk", type=int, default=None, help="partner q-points per batch")


def add_parser(sub):
    import argparse
    a = sub.add_parser("linewidth", help="three-phonon linewidth at arbitrary (off-mesh) q-points",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_linewidth")
    C.add_q_args(a)
    _add_lw_args(a)
    a.add_argument("--bands", default=None, metavar="j1,j2,..", help="band indices (0-based; default all)")
    a.add_argument("--mesh-check", type=int, default=0, metavar="N",
                   help="compare with the kappa-path mesh linewidth at N irreducible points of the config mesh")
    a.set_defaults(func=main_linewidth)
    b = sub.add_parser("attenuation", help="linewidth / attenuation of acoustic modes along directions near Gamma",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(b, "out_attenuation")
    _add_lw_args(b)
    b.add_argument("--directions", default="1,0,0; 1,1,0; 1,1,1", help="Cartesian directions separated by ';'")
    b.add_argument("--qnorm", default=None, metavar="q1,q2,..", help="|q| values, 1/Angstrom (no 2 pi)")
    b.add_argument("--qmin", type=float, default=0.002)
    b.add_argument("--qmax", type=float, default=0.05)
    b.add_argument("--nq", type=int, default=8, help="number of |q| values, logarithmic between --qmin and --qmax")
    b.set_defaults(func=main_attenuation)


def _mesh_of(args, ctx):
    if args.mesh:
        return [int(x) for x in args.mesh.split(",")]
    if ctx.s.used_grg:
        raise SystemExit("[Error] the config mesh is a generalized regular grid; give a diagonal --mesh")
    return [int(x) for x in ctx.s.mesh]


def main_linewidth(args):
    total0 = time.time()
    C.resolve_config(args)
    ctx = C.make_context(args, "linewidth", mesh="config", need_conductivity=bool(args.mesh_check))
    s = ctx.s
    mesh = _mesh_of(args, ctx)
    lw = LinewidthQ(ctx, mesh, args.method, args.sigma, args.chunk)
    log(f"[LW] partner mesh {mesh} ({lw.N} points), method={args.method} sigma={args.sigma}, "
        f"batch={lw._chunk(ctx.nb)} points, tetrahedron vertices: {lw.n_u} distinct offsets")
    chk = {}
    data = dict(C.structure_datasets(s))
    if args.mesh_check:
        c = ctx.c
        gps = s.ir_grid_points
        pick = np.unique(np.linspace(0, len(gps) - 1, min(args.mesh_check, len(gps))).round().astype(int))
        lwm = LinewidthQ(ctx, [int(x) for x in s.mesh], "tetrahedron" if c.sigma is None else "gaussian", c.sigma, args.chunk)
        import phono3py.other.tetrahedron_method as tm
        rel_cfg = np.array(np.dot(tm.get_tetrahedra_relative_grid_address(s.bz_grid.microzone_lattice), s.bz_grid.P.T),
                           dtype="int64")
        chk["mesh_check_tetrahedra_same_as_setup"] = bool(np.array_equal(rel_cfg, lwm.rel))
        t0 = time.time()
        g_ref = np.stack([c.average_by_degeneracy(c.gamma_at_gp(gps[i]), c.freqs[gps[i]].cpu().numpy()).cpu().numpy()
                          for i in pick], axis=1)
        ctx.sync()
        t_ref = time.time() - t0
        t0 = time.time()
        g_new, _, _ = lwm.run(s.qpoints_bz[gps[pick]], progress=False)
        ctx.sync()
        t_new = time.time() - t0
        chk.update({f"mesh_check_t_stage_{k}_s": v for k, v in lwm.t_last.items()})
        big = g_ref > 1e-6 * g_ref.max()
        chk.update(mesh_check_n_q=int(len(pick)), mesh_check_gamma_max_THz=float(g_ref.max()),
                   mesh_check_max_rel=float((np.abs(g_new - g_ref)[big] / g_ref[big]).max()),
                   mesh_check_max_abs_over_max=float(np.abs(g_new - g_ref).max() / g_ref.max()),
                   mesh_check_t_kappa_path_s=t_ref, mesh_check_t_streamed_s=t_new)
        if not args.chunk:
            lws = LinewidthQ(ctx, [int(x) for x in s.mesh], lwm.method, c.sigma, chunk=max(1, lwm.N // 7))
            g_small, _, _ = lws.run(s.qpoints_bz[gps[pick[:2]]], progress=False)
            chk["mesh_check_batch_size_dependence_rel"] = float(
                np.abs(g_small - g_new[:, :2]).max() / np.abs(g_new[:, :2]).max())
        data.update({"mesh_check/grid_point": gps[pick], "mesh_check/qpoint": s.qpoints_bz[gps[pick]],
                     "mesh_check/gamma_kappa_path": g_ref, "mesh_check/gamma_streamed": g_new})
    q, dist, _ = C.parse_qpoints(args, s.lattice)
    if q is not None:
        bands = [int(x) for x in args.bands.split(",")] if args.bands else None
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
        t0 = time.time()
        g, f, bands = lw.run(q, bands)
        ctx.sync()
        chk.update({f"t_stage_{k}_s": v for k, v in lw.t_last.items()})
        chk.update(n_q=int(len(q)), t_linewidth_s=time.time() - t0,
                   t_per_q_s=(time.time() - t0) / len(q), partner_points_per_s=lw.N * len(q) / (time.time() - t0))
        gv = velocities_at(ctx, q)
        data.update(qpoint=q, band_index=np.array(bands), frequency=f, group_velocity=gv, gamma=g,
                    temperature=s.temperatures, mesh=np.array(mesh))
        if dist is not None:
            data["distance"] = dist
    elif not args.mesh_check:
        raise SystemExit("[Error] give --qpoints, --qpoints-file, --path or --mesh-check")
    chk["gpu_peak_GB"] = ctx.peak_gb()
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    units = dict(qpoint="reduced", frequency="THz", gamma="THz", group_velocity="THz*Angstrom", temperature="K")
    C.write_hdf5(f"{args.outdir}/linewidth.hdf5", data, units=units,
                 attrs=dict(checks_json=json.dumps(chk), config=str(args.config), method=args.method,
                            sigma=args.sigma or 0.0))
    log(f"=== [FINISH] analyze linewidth total {time.time() - total0:.2f} s")
    return 0


def main_attenuation(args):
    from phonopy.phonon.degeneracy import degenerate_sets
    total0 = time.time()
    C.resolve_config(args)
    ctx = C.make_context(args, "attenuation", mesh="config", need_conductivity=False)
    s = ctx.s
    mesh = _mesh_of(args, ctx)
    lw = LinewidthQ(ctx, mesh, args.method, args.sigma, args.chunk)
    dirs = np.array([C._vec(t) for t in args.directions.split(";") if t.strip()])
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    qn = (np.array([float(x) for x in args.qnorm.split(",")]) if args.qnorm
          else np.geomspace(args.qmin, args.qmax, args.nq))
    nd, nk, nT = len(dirs), len(qn), len(s.temperatures)
    log(f"[ATT] partner mesh {mesh} ({lw.N} points), method={args.method} sigma={args.sigma}; "
        f"{nd} directions x {nk} |q| values from {qn.min():.3g} to {qn.max():.3g} 1/A")
    qr = ((dirs[:, None, :] * qn[None, :, None]) @ np.asarray(s.lattice).T)  # (nd, nk, 3) reduced
    gam = np.zeros((nT, nd, nk, 3))
    freq = np.zeros((nd, nk, 3))
    vpar = np.zeros((nd, nk, 3))
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(ctx.device)
    t0 = time.time()
    for i in range(nd):
        for k in range(nk):
            g, f = lw.gamma(qr[i, k], [0, 1, 2])
            gam[:, i, k], freq[i, k] = g.cpu().numpy(), f[:3].cpu().numpy()
            _, vop = velocities_at(ctx, qr[i, k][None], f[None], want_vop=True)
            A = np.einsum("jkx,x->jk", vop[0][:3, :3], dirs[i])
            for dset in degenerate_sets(freq[i, k]):
                vpar[i, k, dset] = np.linalg.eigvalsh(A[np.ix_(dset, dset)]) if len(dset) > 1 else A[dset[0], dset[0]].real
            el = time.time() - t0
            done = i * nk + k + 1
            log(f"[ATT] n=[{dirs[i][0]:.3f} {dirs[i][1]:.3f} {dirs[i][2]:.3f}] |q|={qn[k]:.4g}: f = "
                + " ".join(f"{x:.4f}" for x in freq[i, k]) + f" THz, gamma(T={s.temperatures[0]:.0f}) = "
                + " ".join(f"{x:.3e}" for x in gam[0, i, k]) + f" THz ({el:.1f} s, eta {el / done * (nd * nk - done):.1f} s)")
    ctx.sync()
    t_run = time.time() - t0
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 2 * np.pi * gam / vpar[None] * 1e10  # THz / (THz*Angstrom) = 1/Angstrom -> 1/m
        qf = freq[None] / (2 * gam)
    chk = dict(t_run_s=t_run, t_per_q_s=t_run / (nd * nk), partner_points_per_s=lw.N * nd * nk / t_run,
               gpu_peak_GB=ctx.peak_gb())
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    data = dict(C.structure_datasets(s))
    data.update(direction=dirs, qnorm=qn, qpoint=qr, frequency=freq, gamma=gam, velocity_parallel=vpar,
                quality_factor=qf, attenuation_amplitude=alpha, attenuation_dB_per_cm=alpha * 8.685889638 / 100,
                temperature=s.temperatures, mesh=np.array(mesh))
    units = dict(qnorm="1/Angstrom (no 2 pi)", qpoint="reduced", frequency="THz", gamma="THz",
                 velocity_parallel="THz*Angstrom", attenuation_amplitude="1/m (Np/m, amplitude)",
                 attenuation_dB_per_cm="dB/cm", temperature="K")
    C.write_hdf5(f"{args.outdir}/attenuation.hdf5", data, units=units,
                 attrs=dict(checks_json=json.dumps(chk), config=str(args.config), method=args.method,
                            sigma=args.sigma or 0.0))
    log(f"=== [FINISH] analyze attenuation total {time.time() - total0:.2f} s")
    return 0
