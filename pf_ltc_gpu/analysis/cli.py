# -*- coding: utf-8 -*-
"""pf-run-gpu analyze <tool> ...   (also: python -m pf_ltc_gpu.analysis <tool> ...)

Tools (each has its own --help with the options, output datasets and units)
  atom-scatter   atom-resolved ph-ph / isotope linewidths, scattering channels, per-atom kappa attribution
  gruneisen      strain-resolved mode Grueneisen tensors (fc2 + fc3), mean / variance split by band, frequency, atom
  qpoints        frequencies, eigenvectors, group velocities at arbitrary q-points or along a path
  elastic        elastic constants from fc2, Christoffel sound velocities, check against small-q group velocities
  linewidth      three-phonon linewidth of modes at arbitrary (off-mesh) q-points; tetrahedron / Gaussian /
                 Lorentzian energy conservation; partner q-points streamed over a mesh of any density
  attenuation    linewidth / attenuation of the acoustic modes along chosen directions near Gamma
  dos            total and atom-projected phonon DOS (tetrahedron method)
  tmatrix        T-matrix scattering by a point defect (mass and force-constant perturbation)
  rotsum         rotational sum rules / Born-Huang conditions of fc2 (fc3): residuals and optional correction
Every tool except atom-scatter also takes --rotational-sum-rule MODE (default off) to run on corrected force
constants.

Example
  pf-run-gpu analyze atom-scatter --config ltc_config.py --outdir out_atom \
      [--reference ../Si64_nocut/phono3py_disp.yaml] [--compare out_iso/kappa-m6x6x6-rta-wigner-iso.hdf5] \
      [--check-star] [--mem-gb 30]
Writes <outdir>/atom_scatter-m{mesh}[-iso].hdf5.  Analysis only; the kappa path is not touched.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _atom_scatter(args):
    total0 = time.time()
    nthreads = args.threads or int(os.environ.get("PF_GPU_THREADS", "1"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(nthreads)
    import h5py
    import torch

    torch.set_num_threads(nthreads)
    from ..cli import make_fc_builder
    from ..conductivity import GPUConductivity
    from ..setup_cpu import build_setup, load_config, log
    from .atom_scatter import AtomScatter, find_defect

    conf = load_config(args.config)
    tag = ("part" + args.gp_part.replace("/", "of")) if args.gp_part else ("merge" if args.gp_merge else None)
    if tag:  # parallel part jobs must not write the same fc2/fc3 hdf5 files (existing files are only read)
        for key in ("fc2_hdf5", "fc3_hdf5"):
            if conf.get(key) and not Path(conf[key]).exists():
                pth = Path(conf[key]); conf[key] = str(pth.with_name(f"{pth.stem}_{tag}{pth.suffix}"))
    device = args.device or conf.get("device", "cuda:0")
    precision = args.precision or conf.get("precision", "fp64")
    is_isotope = conf.get("is_isotope", True) if args.isotope is None else args.isotope == "on"
    if conf.get("boundary_mfp_um") is not None:
        log("[Warn] boundary scattering is not included in the atom split (not site-resolved).")
    log(f"=== PF-GPU analyze atom-scatter (PID {os.getpid()}) device={device} precision={precision} "
        f"isotope={is_isotope}")
    if device.startswith("cuda"):
        log(f"[GPU] {torch.cuda.get_device_name(torch.device(device))}")
    s = build_setup(conf, fc_builder=make_fc_builder(conf, device))
    c = GPUConductivity(s, device=device, precision=precision, mem_budget_gb=args.mem_gb,
                        chunk_qp=conf.get("chunk_qp"), chunk_band=conf.get("chunk_band"),
                        sigma=conf.get("sigma"),
                        average_degenerate_weights=bool(conf.get("average_degenerate_weights", False)),
                        thm_vertex_limit=bool(conf.get("thm_vertex_limit", False)))
    log(f"[ATOM] n_ir={len(s.ir_grid_points)} nb={c.nb} n_atom={len(s.masses)} "
        f"mem_budget={c.mem_budget / 1e9:.1f} GB")
    min_f = c.min_frequency()
    if min_f < -float(conf.get("neg_tol", 0.1)) and not args.ignore_unstable:
        log(f"UNSTABLE: min_freq={min_f:.4f} THz -> stop (use --ignore-unstable)")
        return 10
    a = AtomScatter(c, fcut=args.channels)
    log(f"[ATOM] triplet chunk = {a._triplet_chunk()}, l block = {a._l_block(a._triplet_chunk())} of {c.nb}")
    t0 = time.time()
    gp_part = tuple(int(v) for v in args.gp_part.split("/")) if args.gp_part else None
    parts = None
    if args.gp_merge:
        import glob as _glob
        parts = sorted(_glob.glob(str(Path(args.gp_merge) / "atom_part-*-of-*.npz")))
        if not parts:
            sys.exit(f"[Error] no atom_part-*.npz in {args.gp_merge}")
    res = a.run(is_isotope=is_isotope, gp_part=gp_part, parts=parts)
    if res.get("partial"):
        os.makedirs(args.outdir, exist_ok=True)
        fn = os.path.join(args.outdir, f"atom_part-{gp_part[0]}-of-{gp_part[1]}.npz")
        np.savez(fn + ".tmp.npz", **{k: v for k, v in res.items() if k != "partial"})
        os.replace(fn + ".tmp.npz", fn)
        log(f"[Save] {fn} ({len(res['indices'])} of {res['n_ir']} grid points, {res['t_gamma']:.1f} s)")
        log(f"=== [FINISH] analyze atom-scatter part total {time.time() - total0:.2f} s")
        return 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_run = time.time() - t0

    # ---- checks
    chk = {}
    g, ga = res["gamma"], res["gamma_atom_sum"]
    big = g > 1e-6 * g.max()
    chk["sumrule_gamma_max_abs_over_max"] = float(np.abs(ga - g).max() / g.max())
    chk["sumrule_gamma_max_rel_per_mode"] = float((np.abs(ga - g)[big] / g[big]).max())
    kP_iso = res["kappa_P_RTA"][:, :3].mean(-1)
    kC_iso = res["kappa_C"][:, :3].mean(-1)
    chk["sumrule_S_P_rel"] = (np.abs(res["S_P_atom"].sum(-1) - kP_iso) / kP_iso).tolist()
    chk["sumrule_S_C_rel"] = (np.abs(res["S_C_atom"].sum(-1) - kC_iso) / np.abs(kC_iso)).tolist()
    if is_isotope:
        chk["sumrule_gamma_iso_check"] = "gamma_isotope is defined as the atom sum"
    if args.compare:
        with h5py.File(args.compare) as h:
            gref = h["gamma"][0]
            chk["vs_main_gamma_max_rel"] = float((np.abs(g - gref)[big] / gref[big]).max())
            chk["vs_main_gamma_atom_sum_max_rel"] = float((np.abs(ga - gref)[big] / gref[big]).max())
            if is_isotope and "gamma_isotope" in h:
                gi = h["gamma_isotope"][()]
                okk = gi > 1e-6 * gi.max()
                chk["vs_main_gamma_iso_max_rel"] = float(
                    (np.abs(res["gamma_isotope"] - gi)[okk] / gi[okk]).max())
            kp = h["kappa_P_RTA"][0]
            kc = h["kappa_C"][0]
            chk["vs_main_kappa_P_max_rel"] = float(np.abs(res["kappa_P_RTA"] - kp)[:, :3].max() / np.abs(kp[:, :3]).max())
            chk["vs_main_kappa_C_max_rel"] = float(np.abs(res["kappa_C"] - kc)[:, :3].max() / np.abs(kc[:, :3]).max())
    if "gamma_channel" in res:
        gc = res["gamma_channel"].sum(-1)
        chk["sumrule_channel_rel"] = float((np.abs(gc - g)[big] / g[big]).max())
        ac = res["gamma_atom_channel"]; nd = ac.shape[-1] // 2  # first half = decay channels
        chk["sumrule_atom_channel_abs_over_max"] = float(np.abs(ac.sum(-1) - res["gamma_atom"]).max() / np.abs(res["gamma_atom"]).max())
        gdec = res["gamma_channel"][..., :6].sum(-1); gdec_atom = ac[..., :nd].sum(-1).sum(-1)
        chk["decay_atom_vs_total_rel"] = float((np.abs(gdec_atom - gdec)[big] / g[big]).max())
        if ac.shape[-1] == 6:  # atom channels summed over atoms = mode channels summed over N/U, class by class
            mc = res["gamma_channel"].reshape(*res["gamma_channel"].shape[:-1], 2, 2, 3).sum(-2).reshape(*g.shape, 6)
            chk["atom_vs_mode_channel_abs_over_max"] = float(np.abs(ac.sum(-2) - mc).max() / g.max())
    if args.check_star:
        t1 = time.time()
        chk["star_check"] = a.star_check(is_isotope)
        chk["star_check_time_s"] = time.time() - t1
    peak = torch.cuda.max_memory_allocated(torch.device(device)) / 1e9 if device.startswith("cuda") else 0.0
    chk.update(t_analysis_s=t_run, t_gamma_s=res["t_gamma"], t_phonon_s=c.t_phonon, gpu_peak_GB=peak)
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")

    # ---- structure + defect
    from phonopy.interface.phonopy_yaml import read_cell_yaml
    cell = s.ph3.primitive
    ref = read_cell_yaml(args.reference) if args.reference else None
    d_idx, d_pos, d_lab = find_defect(cell, ref)
    log(f"[ATOM] defect: {d_lab}")
    mesh = res["mesh"]
    os.makedirs(args.outdir, exist_ok=True)
    fn = os.path.join(args.outdir, f"atom_scatter-m{mesh[0]}x{mesh[1]}x{mesh[2]}" + ("-iso" if is_isotope else "") + ".hdf5")
    with h5py.File(fn, "w") as h:
        for k, v in res.items():
            if k.startswith("t_"):
                continue
            h.create_dataset(k, data=np.asarray(v), compression="gzip" if np.asarray(v).size > 1000 else None)
        h.create_dataset("lattice", data=np.array(cell.cell))
        h.create_dataset("positions", data=np.array(cell.positions))
        h.create_dataset("scaled_positions", data=np.array(cell.scaled_positions))
        h.create_dataset("symbols", data=np.array(list(cell.symbols), dtype="S3"))
        h.create_dataset("masses", data=np.array(cell.masses))
        h.create_dataset("defect_index", data=d_idx)
        h.create_dataset("defect_position", data=np.asarray(d_pos))
        h.attrs["defect_label"] = d_lab
        h.attrs["is_isotope"] = is_isotope
        h.attrs["checks_json"] = json.dumps(chk)
        h.attrs["config"] = str(Path(args.config).resolve())
        h.attrs["note"] = ("gamma_atom[T,q,band,i]: Re(Phi_i^* Phi) split of the ph-ph linewidth (THz); "
                           "S_*_atom: per-atom kappa attribution (W/mK, isotropic trace/3, orbit-averaged); "
                           "heat_capacity computed from frequency (eV/K)")
    log(f"[Save] {fn}")
    log(f"=== [FINISH] analyze atom-scatter total {time.time() - total0:.2f} s")
    return 0


# tools that live in their own module (module name, one-line help); the module provides add_parser(sub) / main(args)
TOOLS = {
    "gruneisen": ("gruneisen", "strain-resolved mode Grueneisen tensors and their mean / variance split"),
    "qpoints": ("qpoints", "frequencies, eigenvectors, group velocities at arbitrary q-points"),
    "elastic": ("elastic", "elastic constants from fc2 and Christoffel sound velocities"),
    "linewidth": ("linewidth_q", "three-phonon linewidth at arbitrary (off-mesh) q-points, streamed partner mesh"),
    "attenuation": ("linewidth_q", "linewidth / attenuation of acoustic modes along directions near Gamma"),
    "dos": ("dos", "total and atom-projected phonon DOS (tetrahedron method)"),
    "tmatrix": ("tmatrix", "T-matrix scattering by a point defect"),
    "rotsum": ("rotsum_tool", "rotational sum rules of the force constants: residuals and optional correction"),
}


def _module_tool(argv):
    """Run one of TOOLS. The thread settings are fixed before torch is imported (as atom-scatter does)."""
    import importlib
    nt = os.environ.get("PF_GPU_THREADS", "1")
    if "--threads" in argv[:-1]:
        nt = argv[argv.index("--threads") + 1]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(nt)
    p = argparse.ArgumentParser(prog="pf-run-gpu analyze")
    sub = p.add_subparsers(dest="tool", required=True)
    for mod in sorted({m for m, _ in TOOLS.values()}):
        try:
            importlib.import_module(f".{mod}", __package__).add_parser(sub)
        except ImportError as e:  # a tool that is not installed must not block the others
            if TOOLS[argv[0]][0] == mod:
                raise
            print(f"[Warn] analysis tool module {mod} not available: {e}")
    args = p.parse_args(argv)
    return args.func(args)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in TOOLS:
        return _module_tool(argv)
    p = argparse.ArgumentParser(prog="pf-run-gpu analyze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="tool", required=True)
    for name, (_, hlp) in TOOLS.items():
        sub.add_parser(name, help=hlp, add_help=False)
    a = sub.add_parser("atom-scatter", help="atom-resolved scattering + per-atom kappa attribution")
    a.add_argument("config", nargs="?", default=None)
    a.add_argument("--config", dest="config_opt", default=None)
    a.add_argument("--device", default=None)
    a.add_argument("--precision", default=None, choices=["fp64", "mixed"])
    a.add_argument("--mem-gb", type=float, default=None)
    a.add_argument("--threads", type=int, default=None)
    a.add_argument("--outdir", default="out_atom")
    a.add_argument("--isotope", choices=["on", "off"], default=None, help="override config is_isotope")
    a.add_argument("--reference", default=None, help="perfect-cell phono3py_disp.yaml (vacancy detection)")
    a.add_argument("--compare", default=None, help="main-path kappa hdf5 of the same case for checks")
    a.add_argument("--check-star", action="store_true", help="verify the symmetry orbit average at one q-star")
    a.add_argument("--ignore-unstable", action="store_true")
    a.add_argument("--gp-part", default=None, metavar="K/M",
                   help="only grid points i %% M == K -> <outdir>/atom_part-K-of-M.npz (run M jobs, then --gp-merge)")
    a.add_argument("--gp-merge", default=None, metavar="DIR", help="read DIR/atom_part-*-of-M.npz and finish")
    a.add_argument("--channels", type=float, default=None, metavar="FCUT_THZ",
                   help="also split the linewidth into decay/coalescence x N/U x partner-frequency classes "
                        "(partners below/above FCUT_THZ); atom-resolved: decay/coalescence x (both below / other)")
    args = p.parse_args(argv)
    args.config = args.config_opt or args.config or "ltc_config.py"
    if not Path(args.config).exists():
        sys.exit(f"[Error] {args.config} not found.")
    if args.tool == "atom-scatter":
        return _atom_scatter(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
