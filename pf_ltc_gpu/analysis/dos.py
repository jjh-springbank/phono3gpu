# -*- coding: utf-8 -*-
"""Total and atom-projected phonon density of states with the linear tetrahedron method.

    pf-run-gpu analyze dos --config ltc_config.py --outdir out_dos
        [--mesh n1,n2,n3]                      (default: mesh of the config)
        [--fmin F --fmax F] [--fpitch P | --fpoints N]   frequency points in THz

Conventions as phonopy (TotalDos / ProjectedDos with the tetrahedron method): the DOS is in states per THz per
primitive cell, sum over all q-points of the regular mesh divided by the number of mesh points, so that the
integral over frequency is 3 * n_atom. Default frequency points as phonopy: from the lowest to the highest
frequency on the mesh in 200 steps (201 points). Tetrahedron weights: kernels.thm_weights_pruned (the 'I'
function of phonopy / phono3py, same as the isotope scattering of the kappa path).

Output <outdir>/dos-m{mesh}.hdf5
  frequency_points (nw,) THz; total_dos (nw,) states/THz/cell; projected_dos (n_atom, nw), sum over atoms = total
  integrated_dos (nw,) cumulative trapezoid integral of total_dos (-> 3 n_atom); mesh (3,)
  projected_dos_xyz (n_atom, 3, nw) split by Cartesian direction of the atomic displacement
Checks: integral of the DOS against 3 n_atom on the given points and on a 10x finer set; sum of the projections.
"""
import json
import time

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import log
from . import common as C


def tetrahedron_dos(c, omega, projected=True, g_chunk=None):
    """(total (nw,), projected (n_p, 3, nw) or None) on the device, for the mesh held by GPUConductivity c."""
    s, dev = c.s, c.device
    nb = c.nb
    n_p = nb // 3
    g = torch.as_tensor(np.array(s.bz_grid.grg2bzg, dtype="int64"), device=dev)
    G = len(g)
    nw = len(omega)
    tot = torch.zeros(nw, device=dev, dtype=torch.float64)
    proj = torch.zeros((n_p, 3, nw), device=dev, dtype=torch.float64) if projected else None
    if g_chunk is None:
        g_chunk = max(1, int(0.25 * c.mem_budget // (nb * (24 * 4 * 8 * 6 + nw * 8 * 3 + nb * 16 * 2))))
    for i in range(0, G, g_chunk):
        gg = g[i:i + g_chunk]
        fv = c.freqs[c.nb_plus[gg]].permute(0, 3, 1, 2)  # (g, nb, 24, 4)
        vs, ci = K.sort_omegas(fv)
        w = K.thm_weights_pruned(omega, vs.reshape(-1, 24, 4), ci.reshape(-1, 24), max_elems=c.thm_elems,
                                 vertex_limit=c.thm_vertex_limit).reshape(nw, len(gg), nb)
        tot += w.sum(dim=(1, 2))
        if projected:
            e2 = (c.eigvecs[gg].abs() ** 2).reshape(len(gg), n_p, 3, nb).to(torch.float64)
            proj += torch.einsum("wgj,giaj->iaw", w, e2)
    return tot / G, (proj / G if projected else None)


def add_parser(sub):
    import argparse
    a = sub.add_parser("dos", help="total and atom-projected phonon DOS (tetrahedron method)",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_dos")
    a.add_argument("--mesh", default=None, metavar="n1,n2,n3")
    a.add_argument("--fmin", type=float, default=None)
    a.add_argument("--fmax", type=float, default=None)
    a.add_argument("--fpitch", type=float, default=None, help="spacing of the frequency points (THz)")
    a.add_argument("--fpoints", type=int, default=None, help="number of frequency points (default 201)")
    a.set_defaults(func=main)


def frequency_points(fmin, fmax, pitch=None, npts=None):
    """phonopy Dos.set_draw_area rule."""
    if pitch is None:
        pitch = (fmax - fmin) / ((npts or 201) - 1)
    return np.arange(fmin, fmax + pitch * 0.1, pitch)


def main(args):
    total0 = time.time()
    C.resolve_config(args)
    mesh = [int(x) for x in args.mesh.split(",")] if args.mesh else "config"
    ctx = C.make_context(args, "dos", mesh=mesh, need_conductivity=True)
    c, s = ctx.c, ctx.s
    g = torch.as_tensor(np.array(s.bz_grid.grg2bzg, dtype="int64"), device=ctx.device)
    fr = c.freqs[g]
    fmin = float(fr.min()) if args.fmin is None else args.fmin
    fmax = float(fr.max()) if args.fmax is None else args.fmax
    pts = frequency_points(fmin, fmax, args.fpitch, args.fpoints)
    t0 = time.time()
    tot, proj = tetrahedron_dos(c, torch.as_tensor(pts, device=ctx.device, dtype=torch.float64))
    ctx.sync()
    t_run = time.time() - t0
    tot, proj = tot.cpu().numpy(), proj.cpu().numpy()
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (tot[1:] + tot[:-1]) * np.diff(pts))])
    fine = np.linspace(float(fr.min()) - 1e-6, float(fr.max()) + 1e-6, 10 * (len(pts) - 1) + 1)
    tf, _ = tetrahedron_dos(c, torch.as_tensor(fine, device=ctx.device, dtype=torch.float64), projected=False)
    tf = tf.cpu().numpy()
    chk = dict(n_modes=int(c.nb), n_frequency_points=int(len(pts)),
               integral_on_points=float(cum[-1]),
               integral_on_10x_finer_points=float(np.sum(0.5 * (tf[1:] + tf[:-1]) * np.diff(fine))),
               projected_sum_vs_total_max_abs=float(np.abs(proj.sum(axis=(0, 1)) - tot).max()),
               dos_max=float(tot.max()), t_dos_s=t_run, gpu_peak_GB=ctx.peak_gb())
    chk["integral_rel_dev_10x"] = abs(chk["integral_on_10x_finer_points"] - c.nb) / c.nb
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    m = np.array(s.bz_grid.D_diag, dtype="int64")
    data = dict(C.structure_datasets(s))
    data.update(frequency_points=pts, total_dos=tot, projected_dos=proj.sum(1), projected_dos_xyz=proj,
                integrated_dos=cum, mesh=m)
    units = dict(frequency_points="THz", total_dos="states/THz per primitive cell",
                 projected_dos="states/THz per primitive cell", projected_dos_xyz="states/THz per primitive cell",
                 integrated_dos="states per primitive cell")
    C.write_hdf5(f"{args.outdir}/dos-m{m[0]}x{m[1]}x{m[2]}.hdf5", data, units=units,
                 attrs=dict(checks_json=json.dumps(chk), config=str(args.config)))
    log(f"=== [FINISH] analyze dos total {time.time() - total0:.2f} s")
    return 0
