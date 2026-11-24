# -*- coding: utf-8 -*-
"""Rotational sum rules / Born-Huang conditions of the force constants: report and optional correction.

    pf-run-gpu analyze rotsum --config ltc_config.py --outdir out_rotsum [--mode born-huang] [--write]

Reports the residuals of the translational, rotational (Born) and Huang conditions of fc2 (with NAC: of the full
analytic dynamical matrix near Gamma), applies the smallest correction that fulfils them (pf_ltc_gpu/rotsum.py),
and reports the change of the force constants. --mode: born | born-huang (default) | born-huang+fc3 (also the
fc3 rotational rule; needs fc2 and fc3 on the same supercell).
--write stores the corrected arrays as <outdir>/fc2_rotsum.hdf5 (and fc3_rotsum.hdf5) in the phono3py format.
Nothing else is changed. To USE the correction in a calculation give the same mode to the run itself:
    pf-run-gpu --config ltc_config.py --rotational-sum-rule born-huang              (kappa)
    pf-run-gpu analyze <tool> ... --rotational-sum-rule born-huang                   (any analysis tool)
or set LTC_CONFIG["rotational_sum_rule"] = "born-huang". Default is off; without the option nothing changes.
Output <outdir>/rotsum.hdf5: attribute report_json; datasets fc2_change (compact, eV/A^2).
"""
import json
import time

import numpy as np

from ..setup_cpu import log
from . import common as C


def add_parser(sub):
    import argparse
    a = sub.add_parser("rotsum", help="rotational sum rules of the force constants: residuals and correction",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_rotsum")
    a.add_argument("--mode", default="born-huang", help="born | born-huang | born-huang+fc3")
    a.add_argument("--write", action="store_true", help="write the corrected force constants to the outdir")
    a.set_defaults(func=main)


def main(args):
    import os
    total0 = time.time()
    C.resolve_config(args)
    args.rotational_sum_rule = None  # this tool applies the correction itself, to report before / after
    from .. import rotsum as RS
    from ..setup_cpu import build_setup, load_config
    from ..cli import make_fc_builder
    conf = load_config(args.config)
    device = args.device or conf.get("device", "cuda:0")
    s = build_setup(conf, mesh_override=[1, 1, 1], fc_builder=make_fc_builder(conf, device))
    ph3 = s.ph3
    conf2 = dict(conf, rotational_sum_rule=args.mode)
    fc2n, fc3n = RS.apply_from_config(conf2, ph3, s.fc2, s.fc3, log=log)
    rep = conf2["_rotsum_report"]
    for part, r in rep.items():
        for k, v in r.items():
            if k != "history":
                log(f"[CHECK] {part}.{k} = {v}")
    data = dict(C.structure_datasets(s))
    data["fc2_change"] = fc2n - s.fc2
    os.makedirs(args.outdir, exist_ok=True)
    if args.write:
        from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
        write_fc2_to_hdf5(fc2n, p2s_map=ph3.phonon_primitive.p2s_map, filename=f"{args.outdir}/fc2_rotsum.hdf5")
        log(f"[Save] {args.outdir}/fc2_rotsum.hdf5")
        if "fc3" in rep:
            write_fc3_to_hdf5(fc3n, p2s_map=ph3.primitive.p2s_map, filename=f"{args.outdir}/fc3_rotsum.hdf5")
            log(f"[Save] {args.outdir}/fc3_rotsum.hdf5")
    C.write_hdf5(f"{args.outdir}/rotsum.hdf5", data, units=dict(fc2_change="eV/Angstrom^2"),
                 attrs=dict(report_json=json.dumps(rep), config=str(args.config), mode=args.mode))
    log(f"=== [FINISH] analyze rotsum total {time.time() - total0:.2f} s")
    return 0
