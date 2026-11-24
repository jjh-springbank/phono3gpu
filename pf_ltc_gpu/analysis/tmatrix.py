# -*- coding: utf-8 -*-
"""Phonon scattering by a point defect: T-matrix from the Green's function of the perfect crystal.

    pf-run-gpu analyze tmatrix --config ltc_config.py --outdir out_tmatrix          (perfect crystal, primitive cell)
        mass defect on one atom:
            --site K (--mass-ratio R | --mass-new M)        K = atom index in the primitive cell, R = M'/M
        defect with changed force constants (substitution), from two supercell calculations with the same cell,
        the same atom order and the same fc2 supercell, one with and one without the defect:
            --defect-config DIR/ltc_config.py --perfect-config DIR/ltc_config.py [--rcut 4.0] [--no-mass] [--no-fc]
        [--mesh n1,n2,n3]        q mesh of the Green's function (default: mesh of the config)
        [--nw 1001]              frequency points of the spectral function for the real part (Kramers-Kronig)
        [--concentration C]      defects per primitive cell (default 1: the rate per unit concentration)
        [--no-star-average]      rates of this one defect at the irreducible q only (no average over the q-star)
        [--device cpu --threads 4]   the same algorithm on CPU threads (timing comparison)

Method (frequencies f in THz; mass-weighted displacement coordinates of the perfect crystal)
    cluster a = (atom, x/y/z) of the atoms touched by the defect; U[a, (q j)] = e_a(q j) exp(2 pi i q.x_atom)
    spectral matrix   rho(f)[a, b] = 1/N sum_qj U U^* delta(f - f_qj)            linear tetrahedron method
    Green's function  g(f^2) = P int df' rho(f') / (f^2 - f'^2)  -  i pi rho(f) / (2 f)
                      real part: rho on a uniform frequency grid (piecewise linear), integrated analytically
                      imaginary part: tetrahedron weights at the frequency of each mode (mean of the two
                      one-sided limits f (1 +- 1e-7), see TMatrix.rates_at_gp)
    perturbation      V(f) = factor^2 dK[a, b] / sqrt(M_a M_b)  -  f^2 (dM_a / M_a) delta_ab
                      dK = fc2(defect cell) - fc2(perfect cell) between cluster atoms, symmetrised, acoustic
                      sum rule restored on the diagonal blocks
    T-matrix          T(f) = (1 - V g)^-1 V
    scattering rate   gamma_def(q j) = - C Im <qj| T(f_qj) |qj> / (2 f_qj)       [THz, as phono3py gamma:
                      lifetime = 1 / (4 pi gamma)], C = defects per primitive cell (dilute limit: linear in C)
    Born approximation (T -> V + V g V) is written as well; for a mass defect it is the isotope (Tamura)
    formula of the kappa path with mass variance C (dM/M)^2 on that atom.
The rates are averaged over the star of each irreducible q-point (= over the symmetry-equivalent orientations
and sites of the defect) unless --no-star-average, and over degenerate modes.

Output <outdir>/tmatrix-m{mesh}.hdf5
  qpoint (n_ir, 3), weight (n_ir,), frequency (n_ir, nb) THz
  gamma_defect (n_ir, nb), gamma_defect_born (n_ir, nb)  THz, for the given concentration
  cluster_atom (n_cl,) primitive-cell atom of each cluster atom; cluster_position (n_cl, 3) reduced coordinates in
  the primitive lattice; delta_mass_ratio (n_cl,) dM/M; perturbation_fc (3 n_cl, 3 n_cl) dK in eV/A^2
  frequency_grid (nw,), cluster_dos (nw,) = trace of rho (states/THz; integral = 3 n_cl)
Checks (attribute checks_json): Born rate against the isotope routine of the kappa path (mass defect), optical
theorem Im<T> = -pi <T rho T>/(2f), sum rule int rho df = 1, Kramers-Kronig real part against the direct sum
above the highest phonon frequency, stage timings.
Limitations of this first version: substitutional defects only (the defect cell must have the same atoms in the
same order as the perfect cell; no vacancy / interstitial), one defect per calculation, no NAC correction of dK,
the mesh is held in memory (not streamed).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import load_config, log
from . import common as C


# ---------------------------------------------------------------------------
# Defect description
# ---------------------------------------------------------------------------
def _load_fc2_case(config_path):
    """(phono3py object, compact fc2) of a case, paths of the config taken relative to the config file."""
    import phono3py
    from phono3py.file_IO import read_fc2_from_hdf5
    conf = load_config(config_path)
    base = Path(config_path).resolve().parent
    yml = base / conf["phono3py_yaml"]
    kw = {"produce_fc": False, "log_level": 0}
    if conf.get("symprec") is not None:
        kw["symprec"] = conf["symprec"]
    if conf.get("primitive_matrix") is not None:
        kw["primitive_matrix"] = conf["primitive_matrix"]
    elif "primitive_matrix" not in yml.read_text():
        kw["primitive_matrix"] = "P"
    cwd = os.getcwd()
    os.chdir(base)  # phono3py.load looks for auxiliary files in the working directory
    try:
        ph3 = phono3py.load(str(yml), **kw)
    finally:
        os.chdir(cwd)
    fc2 = read_fc2_from_hdf5(filename=str(base / conf["fc2_hdf5"]), p2s_map=ph3.phonon_primitive.p2s_map)
    return ph3, np.array(fc2, dtype="double")


class Defect:
    """Cluster of the defect: atoms (primitive index kappa, reduced position x), dM/M, dK (eV/A^2)."""

    def __init__(self, kappa, x, dm_ratio, dK, label):
        self.kappa = np.asarray(kappa, dtype="int64")
        self.x = np.asarray(x, dtype="double")
        self.dm_ratio = np.asarray(dm_ratio, dtype="double")
        self.dK = np.asarray(dK, dtype="double")
        self.label = label
        self.info = {}

    @classmethod
    def mass_defect(cls, s, site, ratio=None, mass_new=None):
        prim = s.ph3.primitive
        m = float(prim.masses[site])
        if ratio is None:
            ratio = mass_new / m
        return cls([site], [np.array(prim.scaled_positions[site])], [ratio - 1.0], np.zeros((3, 3)),
                   f"mass defect on atom {site} ({prim.symbols[site]}), M'/M = {ratio:.6f}")

    @classmethod
    def from_supercells(cls, s, defect_config, perfect_config, rcut=4.0, use_mass=True, use_fc=True, tol=0.7):
        phd, fcd = _load_fc2_case(defect_config)
        php, fcp = _load_fc2_case(perfect_config)
        sd, sp = phd.phonon_supercell, php.phonon_supercell
        if len(sd) != len(sp) or fcd.shape != fcp.shape:
            raise SystemExit("[Error] defect and perfect calculations must have the same atoms / fc2 supercell "
                             "(vacancies and interstitials are not supported)")
        L = np.array(sp.cell)
        if np.abs(np.array(sd.cell) - L).max() > 0.02 * np.abs(L).max():
            raise SystemExit("[Error] the lattices of the defect and the perfect supercell differ by more than 2%")
        fp = np.array(sp.scaled_positions)
        dd = np.array(sd.scaled_positions) - fp
        dd -= np.round(dd)
        disp = np.linalg.norm(dd @ L, axis=1)
        if disp.max() > tol:
            raise SystemExit(f"[Error] atom {disp.argmax()} moved by {disp.max():.2f} A: atom orders do not match")
        md, mp = np.array(sd.masses), np.array(sp.masses)
        pprim = php.phonon_primitive
        p2s, s2p = np.array(pprim.p2s_map), np.array(pprim.s2p_map)
        p2p = pprim.p2p_map
        changed = [i for i in p2s if abs(md[i] - mp[i]) > 1e-6 or sd.symbols[i] != sp.symbols[i]]
        if len(changed) != 1:
            raise SystemExit(f"[Error] expected one substituted atom in the defect cell, found {len(changed)}")
        d0 = changed[0]

        def mic(v):  # minimum-image Cartesian vectors in the supercell
            v = v - np.round(v)
            return v @ L

        r = np.linalg.norm(mic(fp - fp[d0]), axis=1)
        cl = np.where(r <= rcut)[0]
        cl = cl[np.argsort(r[cl], kind="stable")]
        half = 0.5 * min(np.linalg.norm(np.array(pprim.cell), axis=1))
        ncl = len(cl)
        # dK(a, b) = dK_compact[cell atom of a, b shifted by -translation of a]
        dfc = fcd - fcp  # (n_cell, n_super, 3, 3)
        dK = np.zeros((ncl, ncl, 3, 3))
        for ia, a in enumerate(cl):
            p = s2p[a]
            t = fp[a] - fp[p]
            tgt = fp[cl] - t
            d = tgt[:, None, :] - fp[None, :, :]
            d -= np.round(d)
            j = np.abs(d).sum(-1).argmin(axis=1)
            if np.abs(d[np.arange(ncl), j]).max() > 1e-5:
                raise SystemExit("[Error] supercell translation not found (perfect supercell not periodic?)")
            dK[ia] = dfc[p2p[p], j]
        dK = dK.transpose(0, 2, 1, 3).reshape(3 * ncl, 3 * ncl)
        asym = float(np.abs(dK - dK.T).max())
        dK = (dK + dK.T) / 2
        dK4 = dK.reshape(ncl, 3, ncl, 3)
        rowsum = dK4.sum(axis=2)  # (ncl, 3, 3)
        asr_before = float(np.abs(rowsum).max())
        for ia in range(ncl):
            dK4[ia, :, ia, :] -= (rowsum[ia] + rowsum[ia].T) / 2
        dK = dK4.reshape(3 * ncl, 3 * ncl)
        dK = (dK + dK.T) / 2
        # ---- cluster atoms in the primitive cell of the perfect crystal of the main calculation
        prim = s.ph3.primitive
        lat = np.array(prim.cell)
        tau = np.array(prim.scaled_positions)
        xc = (fp[cl] @ L) @ np.linalg.inv(lat)  # reduced coordinates in the primitive lattice
        best = None
        for k0 in range(len(prim)):
            shift = xc[0] - tau[k0]
            shift = shift - np.round(shift)  # origin offset between the two descriptions (fractional part)
            rel = xc[:, None, :] - shift[None, None, :] - tau[None, :, :]
            dev = np.abs(rel - np.round(rel)).max(axis=-1)  # (ncl, n_prim)
            kap = dev.argmin(axis=1)
            worst = dev[np.arange(ncl), kap].max()
            if best is None or worst < best[0]:
                best = (worst, kap, np.round(rel[np.arange(ncl), kap]) + tau[kap])
        worst, kap, x = best
        worst_A = worst * np.linalg.norm(lat, axis=1).max()
        if worst_A > 0.05:
            raise SystemExit(f"[Error] the perfect supercell is not a supercell of the primitive cell of --config "
                             f"(largest mismatch {worst_A:.3f} A)")
        okm = np.abs(mp[cl] - np.array(prim.masses)[kap]).max() < 1e-6
        if not okm:
            raise SystemExit("[Error] species of the perfect supercell do not match the primitive cell")
        ratio = (md[cl] - mp[cl]) / mp[cl] if use_mass else np.zeros(ncl)
        if not use_fc:
            dK = np.zeros_like(dK)
        obj = cls(kap, x, ratio, dK, f"{sd.symbols[d0]} on {sp.symbols[d0]} site, {ncl} atoms within {rcut} A")
        obj.info = dict(n_cluster_atoms=int(ncl), rcut_A=float(rcut), dK_max_eV_A2=float(np.abs(dK).max()),
                        dK_asymmetry_before_symmetrisation=asym, dK_rowsum_before_asr=asr_before,
                        dfc_max_all_pairs_eV_A2=float(np.abs(dfc).max()),
                        lattice_match_A=float(worst_A), max_relaxation_A=float(disp.max()),
                        rcut_exceeds_half_defect_cell=bool(rcut > half))
        return obj


# ---------------------------------------------------------------------------
class TMatrix:
    def __init__(self, ctx, defect, nw=1001):
        self.ctx, self.d = ctx, defect
        c, s, dev = ctx.c, ctx.s, ctx.device
        self.c = c
        self.nb = c.nb
        self.timing = {}
        t0 = time.time()
        g = torch.as_tensor(np.array(s.bz_grid.grg2bzg, dtype="int64"), device=dev)
        self.g = g
        self.N = len(g)
        nb, N = self.nb, self.N
        ncl = len(defect.kappa)
        self.nc = 3 * ncl
        q = c.qbz[g]  # (N, 3)
        x = torch.as_tensor(defect.x, device=dev, dtype=torch.float64)
        ph = torch.exp(2j * np.pi * (q @ x.T))  # (N, ncl)
        rows = torch.as_tensor((defect.kappa[:, None] * 3 + np.arange(3)[None]).reshape(-1), device=dev)
        e = c.eigvecs[g][:, rows, :]  # (N, nc, nb)
        U = e * ph.repeat_interleave(3, dim=1)[:, :, None]
        self.U = U.permute(1, 0, 2).reshape(self.nc, N * nb).contiguous()  # [a, q*nb + j]
        del e, U
        m = torch.as_tensor(np.asarray(s.masses)[defect.kappa], device=dev, dtype=torch.float64).repeat_interleave(3)
        self.VK = (torch.as_tensor(defect.dK, device=dev, dtype=torch.float64) / torch.sqrt(m[:, None] * m[None, :])
                   * s.factor ** 2).to(torch.complex128)
        self.dmr = torch.as_tensor(defect.dm_ratio, device=dev, dtype=torch.float64).repeat_interleave(3)
        # tetrahedron vertices of the whole mesh (as GPUConductivity._isotope_setup)
        fv = c.freqs[c.nb_plus[g]].permute(0, 3, 1, 2)
        vs, ci = K.sort_omegas(fv)
        self.vs, self.ci = vs.reshape(-1, 24, 4), ci.reshape(-1, 24)
        self.fr = c.freqs[g]
        self.Dg = c.degenerate_average_matrices()[g]  # (N, nb, nb)
        self.fmax = float(self.fr.max())
        self.fgrid = torch.linspace(0.0, 1.06 * self.fmax, nw, device=dev, dtype=torch.float64)
        self._tick("setup_U", t0)

    def _tick(self, key, t0):
        self.ctx.sync()
        self.timing[key] = self.timing.get(key, 0.0) + time.time() - t0
        return time.time()

    def weights(self, omega):
        """delta(omega - f_qj) tetrahedron weights (J, N * nb) (sum over q j / N = DOS), averaged over the
        degenerate modes of each q-point: without this average the spectral matrix depends on the arbitrary choice
        of eigenvectors inside degenerate sets (CPU and GPU runs then differ by ~1 %)."""
        c = self.c
        W = K.thm_weights_pruned(omega, self.vs, self.ci, max_elems=c.thm_elems, vertex_limit=c.thm_vertex_limit)
        W = torch.einsum("gjJ,wgJ->wgj", self.Dg, W.reshape(len(omega), self.N, self.nb))
        return W.reshape(len(omega), self.N * self.nb)

    def rho_from_weights(self, W):
        """(J, nc, nc) spectral matrices from weights (J, N * nb); only the non-zero columns are used."""
        out = torch.empty((W.shape[0], self.nc, self.nc), device=W.device, dtype=torch.complex128)
        for k in range(W.shape[0]):
            cols = W[k].nonzero(as_tuple=True)[0]
            Uc = self.U[:, cols]
            out[k] = (Uc * W[k, cols].to(Uc.dtype)[None]) @ Uc.conj().T
        return out / self.N

    def spectral_grid(self, jchunk=64):
        t0 = time.time()
        rho = []
        for i in range(0, len(self.fgrid), jchunk):
            rho.append(self.rho_from_weights(self.weights(self.fgrid[i:i + jchunk])))
        self.rho = torch.cat(rho)  # (nw, nc, nc)
        self._tick("spectral_grid", t0)
        t0 = time.time()
        f = self.fgrid
        nw = len(f)
        dlt = float(f[1] - f[0])
        xm = 0.5 * (f[1:] + f[:-1])  # (nw - 1,) targets
        self.xm = xm
        X = xm[:, None]

        def I0(a, b):
            return (torch.log(torch.abs((X + b) / (X - b))) - torch.log(torch.abs((X + a) / (X - a)))) / (2 * X)

        def I1(a, b):
            return -0.5 * (torch.log(torch.abs(X * X - b * b)) - torch.log(torch.abs(X * X - a * a)))

        a, b = f[None, :-1], f[None, 1:]  # segments
        i0, i1 = I0(a, b), I1(a, b)  # (nw-1 targets, nw-1 segments)
        rise = (i1 - a * i0) / dlt  # weight of the right node of each segment
        fall = (b * i0 - i1) / dlt  # weight of the left node
        Kmat = torch.zeros((nw - 1, nw), device=f.device, dtype=torch.float64)
        Kmat[:, 1:] += rise
        Kmat[:, :-1] += fall
        self.reG = torch.einsum("mk,kab->mab", Kmat.to(torch.complex128), self.rho)  # Hermitian part of g at xm
        self._tick("kramers_kronig", t0)

    def re_g(self, f):
        """Hermitian part of g at the frequencies f (J,) by linear interpolation between the mid-points."""
        xm = self.xm
        p = (f - xm[0]) / (xm[1] - xm[0])
        i0 = torch.clamp(torch.floor(p).to(torch.int64), 0, len(xm) - 2)
        t = (p - i0.to(p.dtype)).to(torch.complex128)[:, None, None]
        return (1 - t) * self.reG[i0] + t * self.reG[i0 + 1]

    def V(self, f):
        """(J, nc, nc) perturbation at the frequencies f (J,)."""
        return self.VK[None] - torch.diag_embed((f[:, None] ** 2 * self.dmr[None, :]).to(torch.complex128))

    def rates_at_gp(self, gp, star=True, at_vertex=False):
        """(gamma_T, gamma_Born, optical-theorem deviation) per unit concentration at BZ grid point gp, (nb,) each.

        The frequency of a mesh mode coincides with tetrahedron vertices (its own and those of the symmetry-
        equivalent q-points, equal up to rounding), where the tetrahedron weight jumps between its one-sided limits
        and zero depending on the last bits (measured: total weight different by ~10 % between a CPU and a GPU run
        at some q-points). The weights are therefore taken as the mean of the two one-sided limits,
        f (1 +- 1e-7). at_vertex=True evaluates exactly at f as the isotope routine of the kappa path does
        (used only for the comparison with that routine)."""
        from phono3py.phonon.grid import get_grid_points_by_rotations
        c, s, dev = self.c, self.ctx.s, self.ctx.device
        nb, nc = self.nb, self.nc
        f0 = c.freqs[gp]
        t0 = time.time()
        # weights averaged over the degenerate sets of the partner modes (in weights()) and of the mode itself,
        # as GPUConductivity.isotope_at_gp does with average_degenerate_weights
        Wr = self.weights(f0) if at_vertex else 0.5 * (self.weights(f0 * (1 + 1e-7)) + self.weights(f0 * (1 - 1e-7)))
        W = torch.einsum("iI,Im->im", self.c.degenerate_average_matrices()[gp], Wr)  # (nb, N * nb)
        del Wr
        rho = self.rho_from_weights(W)  # (nb, nc, nc)
        t0 = self._tick("mode_spectral", t0)
        ok = f0 > max(s.cutoff_frequency, 1e-6)
        fs = torch.where(ok, f0, torch.ones_like(f0))
        img = -np.pi * rho / (2 * fs)[:, None, None]
        g = self.re_g(fs) + 1j * img
        V = self.V(fs)
        eye = torch.eye(nc, device=dev, dtype=torch.complex128)[None]
        T = torch.linalg.solve(eye - V @ g, V)
        t0 = self._tick("mode_solve", t0)
        if star:
            rm = np.unique(get_grid_points_by_rotations(gp, s.bz_grid))
            gr = np.array(s.bz_grid.bzg2grg)[rm]
        else:
            gr = np.array(s.bz_grid.bzg2grg)[[gp]]
        cols = torch.as_tensor(gr[:, None] * nb + np.arange(nb)[None], device=dev)  # (nstar, nb)
        Us = self.U[:, cols]  # (nc, nstar, nb)
        TU = torch.einsum("jab,bsj->asj", T, Us)
        tl = torch.einsum("asj,asj->sj", Us.conj(), TU)  # <qj|T|qj>
        VU = torch.einsum("jab,bsj->asj", V, Us)
        born = torch.einsum("asj,jab,bsj->sj", VU.conj(), 1j * img, VU)  # i Im part of <V g V>
        opt = torch.einsum("asj,jab,bsj->sj", TU.conj(), img.to(torch.complex128), TU).real  # -pi <T rho T>/(2f)
        gT = torch.where(ok, -tl.imag.mean(0) / (2 * fs), torch.zeros_like(fs))
        gB = torch.where(ok, -born.imag.mean(0) / (2 * fs), torch.zeros_like(fs))
        dev_opt = torch.where(ok[None], (tl.imag - opt).abs(), torch.zeros_like(opt)).amax(0)
        scale = torch.where(ok[None], tl.imag.abs(), torch.zeros_like(opt)).amax(0)
        self._tick("mode_rates", t0)
        fh = f0.cpu().numpy()
        return (c.average_by_degeneracy(gT, fh), c.average_by_degeneracy(gB, fh), dev_opt, scale)

    def run(self, star=True, progress=True):
        s = self.ctx.s
        gps = s.ir_grid_points
        n_ir = len(gps)
        gT, gB = np.zeros((n_ir, self.nb)), np.zeros((n_ir, self.nb))
        dopt, sopt = 0.0, 0.0
        t0 = time.time()
        for i, gp in enumerate(gps):
            a, b, d, sc = self.rates_at_gp(gp, star)
            gT[i], gB[i] = a.cpu().numpy(), b.cpu().numpy()
            dopt, sopt = max(dopt, float(d.max())), max(sopt, float(sc.max()))
            if progress and (i % max(1, n_ir // 5) == 0 or i == n_ir - 1):
                log(f"[TMAT] gp {i + 1}/{n_ir} ({time.time() - t0:.1f} s)")
        return gT, gB, dopt / sopt if sopt > 0 else 0.0

    # ------------------------------------------------------------------ checks
    def check_sum_rule(self):
        f = self.fgrid
        integ = (0.5 * (self.rho[1:] + self.rho[:-1]) * (f[1] - f[0])).sum(0)
        eye = torch.eye(self.nc, device=f.device, dtype=torch.complex128)
        return float((integ - eye).abs().max())

    def check_kk(self):
        """Hermitian part of g above the band (exact: plain sum over modes) versus Kramers-Kronig."""
        i = int(torch.searchsorted(self.xm, torch.tensor(1.03 * self.fmax, device=self.xm.device)))
        i = min(i, len(self.xm) - 1)
        x = self.xm[i]
        wgt = (1.0 / (x * x - self.fr.reshape(-1) ** 2)).to(torch.complex128)
        ref = (self.U * wgt[None]) @ self.U.conj().T / self.N
        return float((self.reG[i] - ref).abs().max() / ref.abs().max()), float(x)

    def check_born_isotope(self, n_gp=8):
        """Born rate of a single-atom mass defect versus kernels.isotope_gamma (no star average)."""
        c, s = self.c, self.ctx.s
        if len(self.d.kappa) != 1 or float(self.VK.abs().max()) > 0:
            return None
        c._isotope_setup()
        mv = torch.zeros(len(s.masses), device=self.ctx.device, dtype=torch.float64)
        mv[int(self.d.kappa[0])] = float(self.d.dm_ratio[0]) ** 2
        c.mv = mv
        gps = s.ir_grid_points
        pick = np.unique(np.linspace(0, len(gps) - 1, min(n_gp, len(gps))).round().astype(int))
        err, scale = 0.0, 0.0
        for i in pick:
            ref = c.isotope_at_gp(gps[i])
            _, gB, _, _ = self.rates_at_gp(gps[i], star=False, at_vertex=True)
            err = max(err, float((gB - ref).abs().max()))
            scale = max(scale, float(ref.abs().max()))
        return dict(n_q=int(len(pick)), max_abs_diff_THz=err, max_gamma_THz=scale, max_diff_over_max=err / scale)


def add_parser(sub):
    import argparse
    a = sub.add_parser("tmatrix", help="T-matrix scattering by a point defect",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_tmatrix")
    a.add_argument("--mesh", default=None, metavar="n1,n2,n3")
    a.add_argument("--site", type=int, default=None, help="primitive-cell atom of a mass defect")
    a.add_argument("--mass-ratio", type=float, default=None, help="M'/M of the mass defect")
    a.add_argument("--mass-new", type=float, default=None, help="new mass (amu) of the mass defect")
    a.add_argument("--defect-config", default=None, help="ltc_config.py of the supercell with the defect")
    a.add_argument("--perfect-config", default=None, help="ltc_config.py of the same supercell without the defect")
    a.add_argument("--rcut", type=float, default=4.0, help="cluster radius around the defect (Angstrom)")
    a.add_argument("--no-mass", action="store_true", help="ignore the mass change (force constants only)")
    a.add_argument("--no-fc", action="store_true", help="ignore the force-constant change (mass only)")
    a.add_argument("--nw", type=int, default=1001, help="frequency points of the spectral function")
    a.add_argument("--concentration", type=float, default=1.0, help="defects per primitive cell")
    a.add_argument("--no-star-average", action="store_true")
    a.set_defaults(func=main)


def main(args):
    total0 = time.time()
    C.resolve_config(args)
    mesh = [int(x) for x in args.mesh.split(",")] if args.mesh else "config"
    ctx = C.make_context(args, "tmatrix", mesh=mesh, need_conductivity=True)
    s, c = ctx.s, ctx.c
    if args.defect_config:
        if not args.perfect_config:
            raise SystemExit("[Error] --defect-config needs --perfect-config")
        d = Defect.from_supercells(s, args.defect_config, args.perfect_config, args.rcut,
                                   use_mass=not args.no_mass, use_fc=not args.no_fc)
    elif args.site is not None and (args.mass_ratio is not None or args.mass_new is not None):
        d = Defect.mass_defect(s, args.site, args.mass_ratio, args.mass_new)
    else:
        raise SystemExit("[Error] give --site with --mass-ratio / --mass-new, or --defect-config with --perfect-config")
    log(f"[TMAT] defect: {d.label}; cluster of {len(d.kappa)} atoms ({3 * len(d.kappa)} coordinates); "
        f"mesh {list(s.bz_grid.D_diag)} ({len(s.bz_grid.grg2bzg)} q-points), nw = {args.nw}")
    for k, v in d.info.items():
        log(f"[TMAT] {k} = {v}")
    t_run0 = time.time()
    tm = TMatrix(ctx, d, nw=args.nw)
    tm.spectral_grid()
    gT, gB, opt = tm.run(star=not args.no_star_average)
    ctx.sync()
    t_run = time.time() - t_run0
    timing = dict(tm.timing)  # before the checks, which call the same routines again
    chk = dict(d.info)
    chk["optical_theorem_max_rel_dev"] = opt
    chk["spectral_sum_rule_max_dev"] = tm.check_sum_rule()
    kk, xk = tm.check_kk()
    chk["kramers_kronig_above_band_rel_dev"] = kk
    chk["kramers_kronig_checked_at_THz"] = xk
    bi = tm.check_born_isotope()
    if bi is not None:
        chk["born_vs_isotope_routine"] = bi
    big = gB > 1e-3 * gB.max()
    if big.any():
        r = gT[big] / gB[big]
        chk["tmatrix_over_born_min_max"] = [float(r.min()), float(r.max())]
    chk.update({f"t_{k}_s": v for k, v in timing.items()})
    chk.update(t_phonons_mesh_s=c.t_phonon, t_tmatrix_total_s=t_run, device=str(ctx.device),
               threads=torch.get_num_threads(), gpu_peak_GB=ctx.peak_gb(), n_cluster_coordinates=int(tm.nc),
               n_modes_mesh=int(tm.N * tm.nb), nw=int(args.nw))
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    gps = s.ir_grid_points
    m = np.array(s.bz_grid.D_diag, dtype="int64")
    cc = args.concentration
    data = dict(C.structure_datasets(s))
    data.update(qpoint=s.qpoints_bz[gps], weight=s.grid_weights,
                frequency=c.freqs[torch.as_tensor(gps, device=ctx.device)].cpu().numpy(),
                gamma_defect=gT * cc, gamma_defect_born=gB * cc, concentration=cc, mesh=m,
                cluster_atom=d.kappa, cluster_position=d.x, delta_mass_ratio=d.dm_ratio, perturbation_fc=d.dK,
                frequency_grid=tm.fgrid.cpu().numpy(),
                cluster_dos=torch.diagonal(tm.rho, dim1=1, dim2=2).real.sum(-1).cpu().numpy())
    units = dict(qpoint="reduced", frequency="THz", gamma_defect="THz", gamma_defect_born="THz",
                 concentration="defects per primitive cell", perturbation_fc="eV/Angstrom^2",
                 frequency_grid="THz", cluster_dos="states/THz", cluster_position="reduced (primitive lattice)")
    C.write_hdf5(f"{args.outdir}/tmatrix-m{m[0]}x{m[1]}x{m[2]}.hdf5", data, units=units,
                 attrs=dict(checks_json=json.dumps(chk), config=str(args.config), defect=d.label))
    log(f"=== [FINISH] analyze tmatrix total {time.time() - total0:.2f} s")
    return 0
