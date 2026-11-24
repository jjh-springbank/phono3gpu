# -*- coding: utf-8 -*-
"""Elastic constants from fc2 (method of long waves) and Christoffel sound velocities.

    pf-run-gpu analyze elastic --config ltc_config.py --outdir out_elastic
        [--directions '1,0,0; 1,1,0; 1,1,1; 0.3,0.5,0.8']   propagation directions, Cartesian (default: these four)
        [--qnorm 0.002,0.004]     |q| in 1/Angstrom (no 2 pi) of the small-q phonons used for the consistency check
        [--fd-step 2e-4]          step in 1/Angstrom of the finite differences of D(q) around Gamma

Method.  Along a ray q = t n the dynamical matrix is expanded as D = D0 + t D1 + t^2 D2 (5-point differences).
With P = the three uniform translations (mass-weighted) and Q = 1 - P P^T, the acoustic branches follow from
    Lambda(n) = P^T D2 P - P^T D1 Q (Q D0 Q)^+ Q D1 P          f_j = factor |q| sqrt(eig_j Lambda(n))
(second-order perturbation theory in q, internal relaxation included through the second term).
Elastic constants (Born-Huang, stress-free crystal), from the rays along x, y, z, y+z, x+z, x+y:
    [ab,gd] = P^T D2 P as a quadratic form in n      (ag,bd) = - P^T D1^g Q (Q D0 Q)^+ Q D1^d P
    C_agbd = ([ab,gd] + [bg,ad] - [bd,ag] + (ag,bd)) rho / (2 pi)^2
With NAC the non-analytic (G = 0) dipole term is removed for the elastic constants (constant electric field);
'christoffel_velocity' then uses the full D(q) along each direction, i.e. it includes the piezoelectric
stiffening of the ionic model, and 'christoffel_velocity_from_C' is the constant-field value from C.
Without NAC the two are the same quantity computed twice (checks the assembly of C).

Output <outdir>/elastic.hdf5
  elastic_constants (6, 6) GPa, Voigt order xx yy zz yz xz xy, symmetrised; elastic_tensor (3, 3, 3, 3) GPa as computed
  density kg/m^3; bulk_modulus_voigt GPa
  direction (nd, 3) unit vectors; christoffel_velocity (nd, 3) and christoffel_velocity_from_C (nd, 3), THz*Angstrom
  (x100 = m/s), ascending; christoffel_polarization (nd, 3, 3) [direction, mode, xyz];
  christoffel_group_velocity (nd, 3, 3) from C (per mode; not unique for degenerate transverse modes)
  small_q/qnorm (nq,), small_q/phase_velocity (nd, nq, 3) = f/|q|, small_q/group_velocity_parallel (nd, nq, 3) = v_g.n,
  small_q/group_velocity (nd, nq, 3, 3), small_q/frequency (nd, nq, 3): the three lowest modes at q = |q| n
Checks printed and stored (attribute checks_json): symmetry residuals of C (minor, major, Huang = stress),
acoustic sum rule residuals, change of C with a doubled step, Christoffel versus small-q phase and group velocities.
asr_PD1P_max is the first-order acoustic term P^T D1 P (D units * Angstrom). It vanishes for force constants
that obey rotational invariance; if it does not, the acoustic branches of the given fc2 are not linear at very
small q (f^2 gets a term linear in q) and deviate from the Christoffel values there. The check
frequency_vs_longwave_with_first_order_term compares the small-q frequencies with the expansion that keeps this
term (small_q/frequency_longwave_first_order), which separates the residual of the fc2 from errors of this tool.
"""
import json
import time

import numpy as np
import torch

from ..setup_cpu import log
from . import common as C
from .qpoints import velocities_at

EV_A3_TO_GPA = 160.21766208
AMU_A3_TO_KG_M3 = 1660.5390666
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class LongWave:
    def __init__(self, ctx, h=2e-4):
        self.ctx, self.h = ctx, h
        dev = ctx.device
        dm = ctx.dm
        self.nb = 3 * dm.n_p
        m = torch.as_tensor(ctx.s.dm_masses, device=dev, dtype=torch.float64)
        self.mass = float(m.sum())
        P = torch.zeros((self.nb, 3), device=dev, dtype=torch.complex128)
        for a in range(3):
            P[a::3, a] = torch.sqrt(m / m.sum())
        self.P = P
        self.Q = torch.eye(self.nb, device=dev, dtype=torch.complex128) - P @ P.T

    def nonanalytic(self, qc):
        """G = 0 dipole-dipole term of kernels.DynMat.dipole_dipole at Cartesian q (B, 3), |q| > 0."""
        n = self.ctx.dm.nac
        eps = n["eps"]
        kek = torch.einsum("ba,ac,bc->b", qc, eps, qc)
        kk = qc[:, :, None] * qc[:, None, :] / kek[:, None, None] * torch.exp(-kek / (4 * n["lam"] ** 2))[:, None, None]
        Z = n["Z"]
        dd = torch.einsum("imk,jnl,bmn->bikjl", Z, Z, kk) * n["factor"] * self.ctx.dm.inv_sqrt_mm[None, :, None, :, None]
        return dd.reshape(-1, self.nb, self.nb).to(torch.complex128)

    def D(self, qc, analytic, direction=None):
        """D at Cartesian q (B, 3). analytic: without the G = 0 non-analytic term; else the full matrix, with the
        limit along 'direction' for q = 0."""
        ctx = self.ctx
        dm = ctx.dm
        q = ctx.to_reduced(qc)
        zero = (qc * qc).sum(-1) < 1e-20
        if dm.nac is None:
            return dm(q)
        if analytic:
            D = dm(q)
            nz = ~zero
            if bool(nz.any()):
                D[nz] = D[nz] - self.nonanalytic(qc[nz])
            return D
        D = dm(q)
        if bool(zero.any()):
            d = ctx.to_reduced(direction.reshape(1, 3)).expand(int(zero.sum()), 3)
            D[zero] = dm(q[zero], q_dir=d)
        return D

    def ray(self, n, analytic, h=None):
        """Taylor coefficients (D0, D1, D2) of D(t n) at t = 0 (n Cartesian (3,), t in 1/Angstrom)."""
        h = h or self.h
        n = torch.as_tensor(n, device=self.ctx.device, dtype=torch.float64)
        t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=n.device, dtype=torch.float64) * h
        f = self.D(t[:, None] * n[None, :], analytic, direction=n)
        D1 = (-f[4] + 8 * f[3] - 8 * f[1] + f[0]) / (12 * h)
        D2 = 0.5 * (-f[4] + 16 * f[3] - 30 * f[2] + 16 * f[1] - f[0]) / (12 * h * h)
        return f[2], D1, D2

    def _pinv(self, D0):
        A = self.Q @ D0 @ self.Q
        w, v = torch.linalg.eigh((A + A.conj().T) / 2)
        if self.nb == 3:
            return torch.zeros_like(A)
        keep = w.abs() > 1e-8 * w.abs().max()
        inv = torch.where(keep, 1.0 / torch.where(keep, w, torch.ones_like(w)), torch.zeros_like(w))
        return (v * inv[None].to(v.dtype)) @ v.conj().T

    def christoffel(self, n, analytic=False, h=None):
        """Lambda(n) (3, 3) real symmetric [D units * Angstrom^2] for the unit vector n, and ASR residuals."""
        D0, D1, D2 = self.ray(n, analytic, h)
        P = self.P
        W = self.Q @ D1 @ P
        L = P.T @ D2 @ P - W.conj().T @ self._pinv(D0) @ W
        res = (float((P.T @ D0 @ P).abs().max()), float((P.T @ D1 @ P).abs().max()), float(L.imag.abs().max()))
        L = L.real
        A1 = P.T @ D1 @ P
        self.last_first_order = (A1 + A1.conj().T) / 2  # zero for a rotationally invariant, stress-free fc2
        return (L + L.T) / 2, res

    def velocities(self, L):
        """Sound velocities (3,) THz*Angstrom ascending and polarisations (3 modes, 3) from Lambda."""
        w, u = torch.linalg.eigh(L)
        v = torch.sqrt(torch.abs(w)) * torch.sign(w) * self.ctx.s.factor
        return v.cpu().numpy(), u.T.cpu().numpy()

    def elastic_tensor(self, h=None):
        """C (3, 3, 3, 3) in GPa [C_agbd as index a, g, b, d] and the square brackets s[a, b, g, d]."""
        dev = self.ctx.device
        P = self.P
        eye = torch.eye(3, device=dev, dtype=torch.float64)
        S, W = {}, []
        D0 = None
        for g in range(3):
            D0, D1, D2 = self.ray(eye[g], True, h)
            S[(g, g)] = (P.T @ D2 @ P).real
            W.append(self.Q @ D1 @ P)
        for g in range(3):
            for d in range(g + 1, 3):
                _, _, D2 = self.ray(eye[g] + eye[d], True, h)
                S[(g, d)] = ((P.T @ D2 @ P).real - S[(g, g)] - S[(d, d)]) / 2
                S[(d, g)] = S[(g, d)]
        s = torch.zeros((3, 3, 3, 3), device=dev, dtype=torch.float64)  # [a, b, g, d]
        for (g, d), v in S.items():
            s[:, :, g, d] = (v + v.T) / 2
        Gi = self._pinv(D0)
        T = torch.zeros((3, 3, 3, 3), device=dev, dtype=torch.float64)  # (ag, bd) as [a, g, b, d]
        for g in range(3):
            for d in range(3):
                T[:, g, :, d] = -(W[g].conj().T @ Gi @ W[d]).real
        # C[a, g, b, d] = [ab,gd] + [bg,ad] - [bd,ag] + (ag,bd)
        Cm = torch.einsum("abgd->agbd", s) + torch.einsum("bgad->agbd", s) - torch.einsum("bdag->agbd", s) + T
        conv = self.mass / self.ctx.s.volume / (2 * np.pi) ** 2 * EV_A3_TO_GPA
        return Cm * conv, s * conv, T * conv

    def christoffel_from_C(self, Cm, n):
        """Lambda(n) from the elastic tensor (GPa) -> same units as christoffel()."""
        conv = self.mass / self.ctx.s.volume / (2 * np.pi) ** 2 * EV_A3_TO_GPA
        n = torch.as_tensor(n, device=Cm.device, dtype=torch.float64)
        L = torch.einsum("agbd,g,d->ab", Cm, n, n) / conv
        return (L + L.T) / 2

    def group_velocity_from_C(self, Cm, n, L):
        conv = self.mass / self.ctx.s.volume / (2 * np.pi) ** 2 * EV_A3_TO_GPA
        n = torch.as_tensor(n, device=Cm.device, dtype=torch.float64)
        w, u = torch.linalg.eigh(L)
        A = (Cm + Cm.permute(0, 3, 2, 1)) / conv  # C_agbd + C_adbg
        vg = torch.einsum("aj,bj,agbd,d->jg", u, u, A, n) / (2 * torch.sqrt(torch.abs(w)))[:, None] * self.ctx.s.factor
        return vg.cpu().numpy()


def voigt(Cm):
    """(3,3,3,3) -> (6,6) with the average over the index permutations that are equal for an elastic tensor."""
    out = np.zeros((6, 6))
    for i, (a, g) in enumerate(VOIGT):
        for j, (b, d) in enumerate(VOIGT):
            out[i, j] = (Cm[a, g, b, d] + Cm[g, a, b, d] + Cm[a, g, d, b] + Cm[g, a, d, b]
                         + Cm[b, d, a, g] + Cm[d, b, a, g] + Cm[b, d, g, a] + Cm[d, b, g, a]) / 8
    return out


def unvoigt(c6):
    Cm = np.zeros((3, 3, 3, 3))
    for i, (a, g) in enumerate(VOIGT):
        for j, (b, d) in enumerate(VOIGT):
            Cm[a, g, b, d] = Cm[g, a, b, d] = Cm[a, g, d, b] = Cm[g, a, d, b] = c6[i, j]
    return Cm


def add_parser(sub):
    import argparse
    a = sub.add_parser("elastic", help="elastic constants from fc2 and Christoffel sound velocities",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_elastic")
    a.add_argument("--directions", default="1,0,0; 1,1,0; 1,1,1; 0.3,0.5,0.8",
                   help="propagation directions, Cartesian, separated by ';'")
    a.add_argument("--qnorm", default="0.002,0.004", help="|q| values (1/Angstrom, no 2 pi) of the small-q check")
    a.add_argument("--fd-step", type=float, default=2e-4, help="finite-difference step in 1/Angstrom")
    a.set_defaults(func=main)


def main(args):
    total0 = time.time()
    C.resolve_config(args)
    ctx = C.make_context(args, "elastic", mesh=None, need_conductivity=False)
    s, dev = ctx.s, ctx.device
    lw = LongWave(ctx, h=args.fd_step)
    t0 = time.time()
    Cm, sq, T = lw.elastic_tensor()
    Cm2, _, _ = lw.elastic_tensor(h=2 * args.fd_step)
    Ch = Cm.cpu().numpy()
    c6 = voigt(Ch)
    chk = {}
    scale = float(np.abs(c6).max())
    chk["C_max_GPa"] = scale
    chk["C_change_with_double_step_rel"] = float((Cm2 - Cm).abs().max()) / scale
    chk["C_minor_symmetry_residual_rel"] = float(np.abs(Ch - Ch.transpose(1, 0, 2, 3)).max()) / scale
    chk["C_major_symmetry_residual_rel"] = float(np.abs(Ch - Ch.transpose(2, 3, 0, 1)).max()) / scale
    chk["huang_condition_residual_rel"] = float((sq - sq.permute(2, 3, 0, 1)).abs().max()) / scale
    chk["voigt_symmetrisation_change_rel"] = float(np.abs(unvoigt(c6) - Ch).max()) / scale
    log("[ELASTIC] C (GPa, Voigt xx yy zz yz xz xy):")
    for row in c6:
        log("   " + " ".join(f"{x:10.3f}" for x in row))
    dirs = np.array([C._vec(t) for t in args.directions.split(";") if t.strip()])
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    qn = np.array([float(x) for x in args.qnorm.split(",")])
    nd, nq = len(dirs), len(qn)
    v_full, v_C, v_Csym = np.zeros((nd, 3)), np.zeros((nd, 3)), np.zeros((nd, 3))
    pol, vg_C = np.zeros((nd, 3, 3)), np.zeros((nd, 3, 3))
    asr = [0.0, 0.0, 0.0]
    A1s, Ls = [], []
    c6t = torch.as_tensor(unvoigt(c6), device=dev)
    for i, n in enumerate(dirs):
        L, res = lw.christoffel(n, analytic=False)
        A1s.append(lw.last_first_order); Ls.append(L)
        asr = [max(a, b) for a, b in zip(asr, res)]
        v_full[i], pol[i] = lw.velocities(L)
        LC = lw.christoffel_from_C(Cm, n)
        v_C[i], _ = lw.velocities(LC)
        v_Csym[i], _ = lw.velocities(lw.christoffel_from_C(c6t, n))
        vg_C[i] = lw.group_velocity_from_C(Cm, n, LC)
    chk["asr_PD0P_max"], chk["asr_PD1P_max"], chk["christoffel_imag_max"] = asr
    key = "christoffel_direct_vs_from_C_rel" if ctx.dm.nac is None else "piezo_stiffening_direct_vs_from_C_rel"
    chk[key] = float(np.abs(v_full - v_C).max() / np.abs(v_full).max())
    chk["christoffel_from_symmetrised_C_vs_raw_C_rel"] = float(np.abs(v_Csym - v_C).max() / np.abs(v_C).max())
    # ---- small-q phonons
    qc = (dirs[:, None, :] * qn[None, :, None]).reshape(-1, 3)
    qr = qc @ np.asarray(s.lattice).T
    f = C.phonons_at(ctx, qr, eigvecs=False)
    gv, vop = velocities_at(ctx, qr, f, want_vop=True)
    fh = f.cpu().numpy()[:, :3].reshape(nd, nq, 3)
    gv = gv[:, :3].reshape(nd, nq, 3, 3)
    vph = fh / qn[None, :, None]
    # v_g . n per mode: diagonal of the projected velocity operator; inside a degenerate set its block eigenvalues
    from phonopy.phonon.degeneracy import degenerate_sets
    vpar = np.zeros((nd, nq, 3))
    for i in range(nd):
        for iq in range(nq):
            A = np.einsum("jkx,x->jk", vop[i * nq + iq][:3, :3], dirs[i])
            for dset in degenerate_sets(fh[i, iq]):
                vpar[i, iq, dset] = np.linalg.eigvalsh(A[np.ix_(dset, dset)]) if len(dset) > 1 else A[dset[0], dset[0]].real
    vpar_s = np.sort(vpar, axis=-1)
    for iq in range(nq):
        chk[f"phase_velocity_vs_christoffel_rel_q{qn[iq]:g}"] = float(np.abs(vph[:, iq] - v_full).max() / v_full.max())
        chk[f"group_velocity_parallel_vs_christoffel_rel_q{qn[iq]:g}"] = float(np.abs(vpar_s[:, iq] - v_full).max() / v_full.max())
    # the same small-q frequencies from the long-wave expansion keeping the first-order acoustic term t P^T D1 P
    # (a residual of the force constants); isolates that residual from errors of this tool
    fpred = np.zeros((nd, nq, 3))
    for i in range(nd):
        for iq in range(nq):
            w = torch.linalg.eigvalsh(qn[iq] * A1s[i] + qn[iq] ** 2 * Ls[i].to(torch.complex128))
            fpred[i, iq] = (torch.sqrt(w.abs()) * torch.sign(w) * s.factor).cpu().numpy()
        for iq in range(nq):
            chk[f"frequency_vs_longwave_with_first_order_term_rel_q{qn[iq]:g}"] = max(
                chk.get(f"frequency_vs_longwave_with_first_order_term_rel_q{qn[iq]:g}", 0.0),
                float(np.abs(fpred[i, iq] - fh[i, iq]).max() / np.abs(fh[i, iq]).max()))
    for i, n in enumerate(dirs):
        log(f"[ELASTIC] n = [{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}]  Christoffel v (m/s): "
            + " ".join(f"{100 * x:9.2f}" for x in v_full[i]) + "   f/|q| at smallest q: "
            + " ".join(f"{100 * x:9.2f}" for x in vph[i, 0]))
    ctx.sync()
    chk.update(t_run_s=time.time() - t0, gpu_peak_GB=ctx.peak_gb(), fd_step=args.fd_step)
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    rho = lw.mass / s.volume * AMU_A3_TO_KG_M3
    data = dict(C.structure_datasets(s))
    data.update(elastic_constants=c6, elastic_tensor=Ch, density=rho, bulk_modulus_voigt=float(c6[:3, :3].sum() / 9),
                direction=dirs, christoffel_velocity=v_full, christoffel_velocity_from_C=v_C,
                christoffel_polarization=pol, christoffel_group_velocity=vg_C)
    data.update({"small_q/qnorm": qn, "small_q/frequency": fh, "small_q/frequency_longwave_first_order": fpred, "small_q/phase_velocity": vph,
                 "small_q/group_velocity_parallel": vpar, "small_q/group_velocity": gv})
    units = dict(elastic_constants="GPa", elastic_tensor="GPa", density="kg/m^3", bulk_modulus_voigt="GPa",
                 christoffel_velocity="THz*Angstrom", christoffel_velocity_from_C="THz*Angstrom",
                 christoffel_group_velocity="THz*Angstrom")
    units.update({"small_q/qnorm": "1/Angstrom (no 2 pi)", "small_q/frequency": "THz",
                  "small_q/phase_velocity": "THz*Angstrom", "small_q/group_velocity": "THz*Angstrom",
                  "small_q/group_velocity_parallel": "THz*Angstrom"})
    C.write_hdf5(f"{args.outdir}/elastic.hdf5", data, attrs=dict(checks_json=json.dumps(chk), config=str(args.config),
                                                                   nac=ctx.dm.nac is not None), units=units)
    log(f"=== [FINISH] analyze elastic total {time.time() - total0:.2f} s")
    return 0
