# -*- coding: utf-8 -*-
"""Harmonic phonons at arbitrary q-points: frequencies, eigenvectors, group velocities.

    pf-run-gpu analyze qpoints --config ltc_config.py --outdir out_q
        --qpoints '0,0,0; 0.1,0,0; 1/3,1/3,0' | --qpoints-file q.txt | --path '0,0,0:0.5,0,0.5:0.5,0.25,0.75'
        [--path-npoints 51] [--cartesian] [--eigenvectors] [--velocity-operator] [--mesh-check]

The q-points need not be on any mesh. Group velocities are the diagonal of the velocity operator of the kappa
path (kernels.velocity_operators, phono3py Wigner convention; inside degenerate sets the eigenvalues of the
operator block), so at a mesh point they are the numbers used for kappa. NAC as in the kappa path; exactly at
Gamma no non-analytic term is added, at 0 < |q| < 1e-5 1/A the direction-dependent limit is used.

Output <outdir>/qpoints.hdf5
  qpoint (n, 3) reduced; qpoint_cartesian (n, 3) 1/Angstrom without 2 pi; distance (n,) along the list, same unit
  frequency (n, nb) THz; group_velocity (n, nb, 3) THz*Angstrom (x100 = m/s)
  mode_atom_weight (n, nb, n_atom) |e_i|^2; participation_ratio (n, nb)
  eigenvector (n, nb, nb) complex, columns = modes            (--eigenvectors)
  velocity_operator (n, nb, nb, 3) complex, THz*Angstrom      (--velocity-operator)
--mesh-check: evaluates the irreducible q-points of the config mesh through this route and reports the largest
difference to the mesh arrays of the kappa path (frequency, group velocity; eigenvectors as the weight of each\nmode inside the corresponding degenerate set of the mesh eigenvectors, which must be 1).
"""
import json
import time

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import log
from . import common as C


def velocities_at(ctx, q, f=None, want_vop=False, chunk=None):
    """Group velocities (B, nb, 3) [and velocity operators (B, nb, nb, 3)] at reduced q (B, 3).

    Same steps as GPUConductivity.velocities, for any q."""
    from phonopy.phonon.degeneracy import degenerate_sets
    s = ctx.s
    q = torch.as_tensor(q, device=ctx.device, dtype=torch.float64).reshape(-1, 3)
    if f is None:
        f = C.phonons_at(ctx, q, eigvecs=False)
    if chunk is None:
        chunk = max(1, int(ctx.mem_budget // (ctx.dm.n_p * ctx.dm.n_s * 9 * 16 * 4 * (8 if ctx.dm.nac is not None else 1))))
        chunk = min(chunk, max(1, int(2e8 // (ctx.nb ** 2 * 16 * 12))))
    gvs, vops = [], []
    fh = f.cpu().numpy()
    for i in range(0, len(q), chunk):
        vop = K.velocity_operators(ctx.dm, q[i:i + chunk], s.lattice, s.factor, dynmat0=ctx.dm0)
        vh = vop.cpu().numpy()
        gv = np.einsum("niix->nix", vh).real.copy()
        for n in range(len(vh)):
            pos = 0
            for deg in degenerate_sets(fh[i + n]):
                if len(deg) > 1:
                    blk = vh[n, pos:pos + len(deg), pos:pos + len(deg), :]
                    for d in range(3):
                        gv[n, pos:pos + len(deg), d] = np.linalg.eigvalsh(blk[..., d])
                pos += len(deg)
        gvs.append(gv)
        if want_vop:
            vops.append(vh)
    gv = np.concatenate(gvs)
    return (gv, np.concatenate(vops)) if want_vop else gv


def add_parser(sub):
    import argparse
    a = sub.add_parser("qpoints", help="frequencies, eigenvectors, group velocities at arbitrary q-points",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_qpoints")
    C.add_q_args(a)
    a.add_argument("--eigenvectors", action="store_true", help="write the eigenvectors")
    a.add_argument("--velocity-operator", action="store_true", help="write the full velocity operators")
    a.add_argument("--mesh-check", action="store_true", help="compare with the mesh arrays of the kappa path")
    a.set_defaults(func=main)


def main(args):
    total0 = time.time()
    C.resolve_config(args)
    ctx = C.make_context(args, "qpoints", mesh="config" if args.mesh_check else None,
                         need_conductivity=args.mesh_check)
    s = ctx.s
    q, dist, seg = C.parse_qpoints(args, s.lattice)
    chk = {}
    if args.mesh_check:
        c = ctx.c
        gps = s.ir_grid_points
        gt = torch.as_tensor(gps, device=ctx.device)
        qm = s.qpoints_bz[gps]
        f, e = C.phonons_at(ctx, qm)
        gv = velocities_at(ctx, qm, f)
        _, gv_ref = c.velocities(gps)
        chk["mesh_check_n_q"] = int(len(gps))
        chk["mesh_check_frequency_max_abs_THz"] = float((f - c.freqs[gt]).abs().max())
        # eigenvectors: weight of each mode inside the matching degenerate set of the mesh eigenvectors (1 = same space)
        ov = torch.einsum("bIj,bIk->bjk", c.eigvecs[gt].conj(), e).abs() ** 2
        inset = (ov * C.degenerate_mask(c.freqs[gt]).to(ov.dtype)).sum(1)
        chk["mesh_check_eigenvector_subspace_max_dev"] = float((inset - 1).abs().max())
        chk["mesh_check_group_velocity_max_abs_THzA"] = float(np.abs(gv - gv_ref.cpu().numpy()).max())
        chk["mesh_check_group_velocity_scale_THzA"] = float(np.abs(gv).max())
        if q is None:
            q, dist = qm, None
    if q is None:
        raise SystemExit("[Error] give --qpoints, --qpoints-file, --path or --mesh-check")
    t0 = time.time()
    f, e = C.phonons_at(ctx, q)
    out = velocities_at(ctx, q, f, want_vop=args.velocity_operator)
    gv = out[0] if args.velocity_operator else out
    ctx.sync()
    t_run = time.time() - t0
    n = len(q)
    wgt = (e.abs() ** 2).reshape(n, ctx.n_p, 3, ctx.nb).sum(2).permute(0, 2, 1)
    data = dict(C.structure_datasets(s))
    data.update(qpoint=q, qpoint_cartesian=q @ np.linalg.inv(s.lattice).T, frequency=f.cpu().numpy(),
                group_velocity=gv, mode_atom_weight=wgt.cpu().numpy(),
                participation_ratio=(1.0 / (ctx.n_p * (wgt ** 2).sum(-1))).cpu().numpy())
    if dist is not None:
        data["distance"] = dist
    if seg is not None:
        data["path_segment_start"] = seg
    if args.eigenvectors:
        data["eigenvector"] = e.cpu().numpy()
    if args.velocity_operator:
        data["velocity_operator"] = out[1]
    chk.update(n_q=int(n), t_run_s=t_run, gpu_peak_GB=ctx.peak_gb())
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    units = dict(qpoint="reduced", qpoint_cartesian="1/Angstrom (no 2 pi)", distance="1/Angstrom (no 2 pi)",
                 frequency="THz", group_velocity="THz*Angstrom", velocity_operator="THz*Angstrom")
    C.write_hdf5(f"{args.outdir}/qpoints.hdf5", data, attrs=dict(checks_json=json.dumps(chk), config=str(args.config)),
                 units=units)
    log(f"=== [FINISH] analyze qpoints total {time.time() - total0:.2f} s")
    return 0
