# -*- coding: utf-8 -*-
"""pf-run-gpu: GPU replacement for the phono3py BTE part of pf-run.

Usage:  pf-run-gpu --config ltc_config.py [--device cuda:0]

Reads phono3py_disp.yaml + fc2.hdf5 + fc3.hdf5 (no fc solving here) and writes
kappa-m{a}x{b}x{c}-rta-wigner-iso.hdf5 in the same format as pf-run.
Exit code 10 = dynamically UNSTABLE (no kappa written), same as precheck_stable.py.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _normalize_hdf5_compression(value, default="gzip"):
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "false", "off", "0"}:
        return None
    return value


def main(argv=None):
    if (sys.argv[1:] if argv is None else argv)[:1] == ["analyze"]:  # analysis tools (separate module)
        from .analysis.cli import main as analyze_main
        return analyze_main((sys.argv[1:] if argv is None else argv)[1:])
    parser = argparse.ArgumentParser(prog="pf-run-gpu")
    parser.add_argument("--config", default="ltc_config.py")
    parser.add_argument("--device", default=None, help="cuda:0, cpu, ... (overrides config)")
    parser.add_argument("--precision", default=None, choices=["fp64", "mixed"])
    parser.add_argument("--mem-gb", type=float, default=None,
                        help="memory budget per block (default: 60%% of free GPU memory)")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads (default 1; GPU runs need only one)")
    parser.add_argument("--outdir", default=".", help="where to write kappa-*.hdf5")
    parser.add_argument("--ignore-unstable", action="store_true",
                        help="compute kappa even with imaginary modes (validation only)")
    parser.add_argument("--gp-part", default=None, metavar="K/M",
                        help="RTA only: compute the linewidths at irreducible grid points i with i %% M == K and "
                             "write <outdir>/gamma_part-K-of-M.npz (no kappa); run M such jobs, then --gp-merge")
    parser.add_argument("--gp-merge", default=None, metavar="DIR",
                        help="RTA only: read DIR/gamma_part-*-of-M.npz for all grid points and compute kappa")
    parser.add_argument("--rotational-sum-rule", default=None, metavar="MODE",
                        help="optional (default off): impose the rotational sum rules on the force constants before "
                             "the calculation; MODE = born | born-huang | born-huang+fc3 (see pf_ltc_gpu/rotsum.py; "
                             "same as the config key rotational_sum_rule)")
    args = parser.parse_args(argv)

    total_start = time.time()
    if not Path(args.config).exists():
        sys.exit(f"[Error] {args.config} not found.")

    # GPU-only policy: keep host threading to a single core unless asked.
    nthreads = args.threads or int(os.environ.get("PF_GPU_THREADS", "1"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(nthreads)

    import torch

    torch.set_num_threads(nthreads)
    from .conductivity import GPUConductivity
    from .setup_cpu import build_setup, load_config, log

    conf = load_config(args.config)
    if args.rotational_sum_rule:
        conf["rotational_sum_rule"] = args.rotational_sum_rule
    tag = None
    if args.gp_part:
        tag = "part" + args.gp_part.replace("/", "of")
    elif args.gp_merge:
        tag = "merge"
    if tag:  # parallel part jobs must not write the same fc2/fc3 hdf5 files (existing files are only read)
        for key in ("fc2_hdf5", "fc3_hdf5"):
            if conf.get(key) and not Path(conf[key]).exists():
                pth = Path(conf[key]); conf[key] = str(pth.with_name(f"{pth.stem}_{tag}{pth.suffix}"))
    device = args.device or conf.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"[Error] device={device} requested but CUDA is not available.")
    precision = args.precision or conf.get("precision", "fp64")

    log(f"=== PhonoFlow LTC Runner GPU (PID: {os.getpid()}) ===")
    log(f"[GPU] device={device} precision={precision} torch={torch.__version__} "
        f"host_threads={nthreads}")
    if device.startswith("cuda"):
        log(f"[GPU] {torch.cuda.get_device_name(torch.device(device))}")

    mode = conf.get("calc_mode", {"base_method": "RTA", "wigner_correction": True})
    base_method = mode["base_method"].upper()
    use_wigner = bool(mode.get("wigner_correction", True))
    if base_method not in ("RTA", "LBTE"):
        sys.exit(f"[Error] base_method={base_method} not supported (RTA or LBTE).")
    if not use_wigner:
        log("[Warn] wigner_correction=False: kappa_C is still computed; kappa = kappa_P_RTA + kappa_C.")
    is_isotope = conf.get("is_isotope", True)
    b_mfp_um = conf.get("boundary_mfp_um", None)
    # pf-run passes boundary_mfp_um * 1e4 to phono3py (which expects micrometre);
    # keep that behaviour for identical numbers.
    boundary_mfp_val = None if b_mfp_um is None else b_mfp_um * 1e4

    t0 = time.time()
    s = build_setup(conf, fc_builder=make_fc_builder(conf, device))
    if s is None:
        log("[STOP] stop_after_force_constants=True; skipping thermal conductivity.")
        log(f"=== [FINISH] Total Piepline Time: {time.time() - total_start:.2f} seconds")
        return 0
    log(f" >>> Setup (CPU, grid/triplets) time: {time.time() - t0:.2f} seconds")
    mesh = s.mesh

    start_ltc = time.time()
    c = GPUConductivity(s, device=device, precision=precision, mem_budget_gb=args.mem_gb,
                        chunk_qp=conf.get("chunk_qp"), chunk_band=conf.get("chunk_band"),
                        sigma=conf.get("sigma"),
                        average_degenerate_weights=bool(conf.get("average_degenerate_weights", False)),
                        thm_vertex_limit=bool(conf.get("thm_vertex_limit", False)))
    log(f" >>> Phonon solve time: {c.t_phonon:.2f} seconds | n_bz={len(s.qpoints_bz)} "
        f"n_ir={len(s.ir_grid_points)} nb={c.nb}")
    log(f" >>> IFC3 formatting time (excluded from compute): {c.t_fc3_prep:.2f} seconds")

    neg_tol = float(conf.get("neg_tol", os.environ.get("NEG_TOL", "0.1")))
    min_f = c.min_frequency()
    if min_f < -neg_tol:
        log(f"UNSTABLE: min_freq={min_f:.4f} THz (tol=-{neg_tol}) "
            f"mesh={mesh[0]}x{mesh[1]}x{mesh[2]}")
        if not args.ignore_unstable:
            log(f"=== [FINISH] Total Piepline Time: {time.time() - total_start:.2f} seconds")
            return 10
        log("[Warn] --ignore-unstable: continuing (negative modes excluded by cutoff).")
    else:
        log(f"STABLE: min_freq={min_f:.4f} THz (tol=-{neg_tol}) "
            f"mesh={mesh[0]}x{mesh[1]}x{mesh[2]}")

    log(f"[LTC] Base: {base_method} (k_p) | Wigner: {use_wigner} (k_c)")
    log(f"      Options: Isotope={is_isotope}, B_MFP={b_mfp_um}um")
    t0 = time.time()
    gp_part = tuple(int(v) for v in args.gp_part.split("/")) if args.gp_part else None
    gamma_parts = None
    if args.gp_merge:
        import glob as _glob
        gamma_parts = sorted(_glob.glob(str(Path(args.gp_merge) / "gamma_part-*-of-*.npz")))
        if not gamma_parts:
            sys.exit(f"[Error] no gamma_part-*.npz in {args.gp_merge}")
    res = c.run(is_isotope=is_isotope, boundary_mfp=boundary_mfp_val, lbte=(base_method == "LBTE"),
                pinv_cutoff=float(conf.get("pinv_cutoff", 1e-8)),
                pinv_method=int(conf.get("lbte_pinv_method", 0)), gp_part=gp_part, gamma_parts=gamma_parts)
    if res.get("partial"):
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        fn = Path(args.outdir) / f"gamma_part-{gp_part[0]}-of-{gp_part[1]}.npz"
        np.savez(str(fn) + ".tmp.npz", **{k: v for k, v in res.items() if k != "partial"})
        os.replace(str(fn) + ".tmp.npz", fn)
        log(f"[Save] {fn} ({len(res['indices'])} of {res['n_ir']} grid points, {res['t_gamma']:.1f} s)")
        log(f"\n=== [FINISH] Total Piepline Time: {time.time() - total_start:.2f} seconds")
        return 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_rtc = time.time() - t0
    log(f" >>> run_thermal_conductivity() time: {t_rtc:.2f} seconds")
    log(f" >>> LTC Calculation Time: {time.time() - start_ltc:.2f} seconds")
    log(f" >>> Compute time (phonon solve + run_thermal_conductivity, no IFC formatting/IO): "
        f"{c.t_phonon + t_rtc:.2f} seconds")
    if device.startswith("cuda"):
        log(f"[GPU] peak memory: {torch.cuda.max_memory_allocated(torch.device(device)) / 1e9:.2f} GB")

    for k, T in enumerate(res["temperature"]):
        kp, kc = res["kappa_P_RTA"][0, k], res["kappa_C"][0, k]
        log(f"[Kappa] T={T:.0f} K  kP(xx,yy,zz)=({kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f})  "
            f"kC=({kc[0]:.3f}, {kc[1]:.3f}, {kc[2]:.3f})  "
            f"k_iso_avg={(kp[:3] + kc[:3]).mean():.3f} W/mK")
        if "kappa_P_exact" in res:
            ke = res["kappa_P_exact"][0, k]
            log(f"[Kappa] T={T:.0f} K  LBTE kP_exact(xx,yy,zz)=({ke[0]:.3f}, {ke[1]:.3f}, {ke[2]:.3f})")
    if "t_lbte_solve" in res:
        log(f" >>> LBTE collision-matrix solve time: {res['t_lbte_solve']:.2f} seconds")

    out_filename = write_kappa(res, conf, mesh, is_isotope, outdir=args.outdir)
    log(f"[Save] {out_filename}")
    log(f"\n=== [FINISH] Total Piepline Time: {time.time() - total_start:.2f} seconds")
    return 0


def make_fc_builder(conf, device):
    """fc2/fc3 from the force mmaps on ``device`` when the hdf5 files are missing
    (same switch, file names and hdf5 output as pf-run)."""

    def build(kind, ph3):
        import torch
        from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

        from .fc_gpu import produce_fc2_compact, produce_fc3_compact
        from .setup_cpu import log

        dt = np.dtype(conf.get(f"{kind}_mmap_dtype", "float32"))
        if kind == "fc2":
            n_snap, nat = len(ph3.phonon_supercells_with_displacements), len(ph3.phonon_supercell)
        else:
            n_snap, nat = len(ph3.supercells_with_displacements), len(ph3.supercell)
        log(f"[{kind.upper()}] Computing from mmap on {device}...")
        mm = np.memmap(conf[f"{kind}_mmap"], dtype=dt, mode="r", shape=(n_snap, nat, 3))
        t0 = time.time()
        if kind == "fc2":
            fc = produce_fc2_compact(ph3, mm, device, symmetrize=conf.get("symmetrize_fc2", True))
        else:
            if conf.get("symmetrize_fc3", True) and not conf.get("fast_symmetrize_fc3", False):
                raise SystemExit("[Error] symmetrize_fc3 without fast_symmetrize_fc3 "
                                 "(produce_fc3(symmetrize_fc3r=True)) is not ported.")
            # fc3_fast_sym: "fixed" (default) = the corrected pf-run step: nothing for a
            # cut-off dataset (already permutation-symmetric), phono3py's compact
            # permutation symmetrisation otherwise.  "pfrun" reproduces the pre-2026-10-13
            # pf-run bug (permutation_symmetry_fc3 on the compact array), "off" skips it.
            mode = conf.get("fc3_fast_sym", "fixed")
            fast = conf.get("fast_symmetrize_fc3", False) and conf.get("symmetrize_fc3", True)
            if fast and mode == "pfrun":
                log("[FC3] fc3_fast_sym='pfrun': reproducing the old pf-run bug "
                    "(permutation_symmetry_fc3 on the compact array)")
            fc = produce_fc3_compact(ph3, mm, device, fast_sym_compat=fast and mode == "pfrun")
            if fast and mode == "fixed":
                if "cutoff_distance" in ph3.dataset:
                    log("[FC3] compact fc3 from a cut-off dataset is already "
                        "permutation-symmetric; skipping fast symmetrization.")
                else:
                    from phono3py.phonon3.fc3 import set_permutation_symmetry_compact_fc3

                    set_permutation_symmetry_compact_fc3(fc, ph3.primitive)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            log(f"[GPU] {kind} peak memory: "
                f"{torch.cuda.max_memory_allocated(torch.device(device)) / 1e9:.2f} GB")
            torch.cuda.empty_cache()  # leave the memory to the BTE budget
        log(f" >>> {kind.upper()} solve: {time.time() - t0:.2f} seconds")
        t0 = time.time()
        comp = _normalize_hdf5_compression(
            conf.get(f"{kind}_hdf5_compression", "gzip" if kind == "fc2" else None),
            default="gzip" if kind == "fc2" else None)
        if kind == "fc2":
            write_fc2_to_hdf5(fc, p2s_map=ph3.phonon_primitive.p2s_map,
                              filename=conf["fc2_hdf5"], compression=comp)
        else:
            write_fc3_to_hdf5(fc, p2s_map=ph3.primitive.p2s_map,
                              filename=conf["fc3_hdf5"], compression=comp)
        log(f" >>> {kind.upper()} HDF5 write: {time.time() - t0:.2f} seconds | compression={comp}")
        return fc

    return build


def write_kappa(res, conf, mesh, is_isotope, outdir="."):
    from phono3py.file_IO import write_kappa_to_hdf5

    method = "lbte" if "kappa_P_exact" in res else "rta"
    base_name = f"kappa-m{mesh[0]}x{mesh[1]}x{mesh[2]}-{method}-wigner"
    if is_isotope:
        base_name += "-iso"
    out_filename = os.path.join(outdir, f"{base_name}.hdf5")
    nT, n_q, nb = len(res["temperature"]), res["frequency"].shape[0], res["frequency"].shape[1]
    compression = _normalize_hdf5_compression(conf.get("kappa_hdf5_compression", "gzip"))
    os.makedirs(outdir, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(outdir)
        write_kappa_to_hdf5(
            gamma=res["gamma"],
            temperature=res["temperature"],
            mesh=res["mesh"],
            frequency=res["frequency"],
            group_velocity=res["group_velocity"],
            heat_capacity=res.get("heat_capacity", np.zeros((nT, n_q, nb), dtype="double")),
            kappa=res["kappa_TOT_exact"] if "kappa_P_exact" in res else res["kappa"],
            kappa_P_exact=res.get("kappa_P_exact"),
            kappa_TOT_exact=res.get("kappa_TOT_exact"),
            kappa_TOT_RTA=res["kappa"] if "kappa_P_exact" in res else None,
            kappa_P_RTA=res["kappa_P_RTA"],
            kappa_C=res["kappa_C"],
            gamma_isotope=res["gamma_iso"] if is_isotope else None,
            qpoint=res["qpoint"],
            weight=res["weight"],
            filename=None,
            compression=compression,
            verbose=False,
        )
        produced = "kappa" + "-m%d%d%d" % tuple(res["mesh"]) + ".hdf5"
        final = os.path.basename(out_filename)
        if produced != final:
            shutil.move(produced, final)
    finally:
        os.chdir(cwd)
    return out_filename


if __name__ == "__main__":
    sys.exit(main())
