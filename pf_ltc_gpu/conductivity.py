# -*- coding: utf-8 -*-
"""RTA + Wigner (+ isotope) lattice thermal conductivity on torch.

Mirrors phono3py.conductivity.rta.ConductivityWignerRTA with the low-memory
pp_collision path (tetrahedron method), sigma=None, is_kappa_star=True.
"""
import time

import numpy as np
import torch
from phonopy import units as U
from phonopy.phonon.degeneracy import degenerate_sets

from . import kernels as K
from .setup_cpu import LTCSetup, kstar_info, log, triplets_at_gp


def _bytes_per_triplet_thm(nb):
    # sorted vertices (3, nb1, nb2, 24, 4) + dense weights (3, nb0, nb1, nb2) and
    # the active mask; thm_weight_I itself is chunked separately.
    return nb ** 2 * 3 * 24 * 4 * 8 * 3 + nb ** 3 * 3 * 8 * 4


def _bytes_per_triplet_pp(n_p, n_s, nb, csize):
    # A1..A3, F, contraction temps (complex) + pp, g1, g2, P1, P2 (float64).
    # The fc3 x phase intermediate X is blocked separately (Fc3Normal.x_budget).
    return 8 * nb ** 3 * csize + 10 * nb ** 3 * 8


class GPUConductivity:
    def __init__(self, s: LTCSetup, device="cpu", precision="fp64",
                 mem_budget_gb=None, chunk_qp=None, chunk_band=None, sigma=None,
                 average_degenerate_weights=False, thm_vertex_limit=False):
        self.s = s
        # phono3py >= 4.8 option: average integration weights over degenerate bands
        # so that Gamma does not depend on the eigenvector gauge (off = 3.12.1 behaviour)
        self.average_degenerate_weights = average_degenerate_weights
        # phono3py >= 4.7 (phonors) rule for omega exactly at a tetrahedron vertex
        self.thm_vertex_limit = thm_vertex_limit
        self._deg = None
        self.device = torch.device(device)
        self.sigma = sigma
        self.cdtype = torch.complex64 if precision == "mixed" else torch.complex128
        if mem_budget_gb is None:
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                mem_budget_gb = 0.6 * free / 1e9
            else:
                mem_budget_gb = 1.0
        self.mem_budget = mem_budget_gb * 1e9
        self.chunk_qp = chunk_qp
        self.chunk_band = chunk_band
        dev = self.device
        self.nb = 3 * len(s.masses)
        self.temps = torch.as_tensor(s.temperatures, device=dev, dtype=torch.float64)

        t0 = time.time()
        nac = s.extra.get("nac")
        if nac is not None:
            self.dm = K.DynMat(nac["fc_short_range"], s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev,
                               nac=nac)
            self.dm0 = K.DynMat(s.fc2, s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev)
        else:
            self.dm = K.DynMat(s.fc2, s.dm_svecs, s.dm_multi, s.dm_masses, s.dm_s2pp, dev)
            self.dm0 = None
        qbz = torch.as_tensor(s.qpoints_bz, device=dev, dtype=torch.float64)
        self.freqs, self.eigvecs = K.solve_phonons(self.dm, qbz, s.factor,
                                                   mem_budget=self.mem_budget / 2)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.t_phonon = time.time() - t0

        # IFC3 formatting (supercell-atom reordering, upload): timed separately and
        # excluded from the compute time used in benchmarks.
        t0 = time.time()
        self.fn = K.Fc3Normal(s.fc3, s.svecs, s.multi, s.masses, s.p2s, s.s2pp,
                              s.all_shortest, s.make_r0_average, s.cutoff_frequency,
                              dev, cdtype=self.cdtype, x_budget=self.mem_budget / 4,
                              band_block=chunk_band)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.t_fc3_prep = time.time() - t0
        # thm_weight_I keeps ~40 live temporaries of this many float64 elements
        self.thm_elems = max(2 ** 20, int(self.mem_budget / 4 // (40 * 8)))
        self.nb_plus = torch.as_tensor(s.nb_plus, device=dev)
        self.nb_minus = torch.as_tensor(s.nb_minus, device=dev)
        self.qbz = qbz

    # ------------------------------------------------------------------
    def degenerate_average_matrices(self):
        """(n_bz, nb, nb) block matrices that average over each degenerate set.

        Degenerate sets follow phonopy's get_degenerate_ids (consecutive bands closer
        than 1e-4 THz), as used by phono3py >= 4.8. Built lazily from self.freqs.
        """
        if self._deg is None:
            self._deg = K.degenerate_average_matrix(K.degenerate_ids(self.freqs))
        return self._deg

    def min_frequency(self):
        return float(self.freqs.min())

    def _triplet_chunk(self):
        if self.chunk_qp is not None:
            return int(self.chunk_qp)
        s = self.s
        csize = 8 if self.cdtype == torch.complex64 else 16
        per = _bytes_per_triplet_pp(len(s.masses), s.fc3.shape[1], self.nb, csize)
        if self.sigma is None:
            per = max(per, _bytes_per_triplet_thm(self.nb))
        return max(1, int(self.mem_budget // per))

    # ------------------------------------------------------------------
    def gamma_at_gp(self, gp, collision=False):
        """Imaginary self-energy Gamma (nT, nb) at BZ grid point gp (ph-ph only).

        collision=True also returns, per irreducible triplet, the two LBTE collision
        sums (phono3py collision_matrix.c, tp_type=3 weights g3):
          A[t, T, j0, j1] = sum_j2 |Phi|^2 g3 / sinh(f2/2kT)   (q1 is the partner)
          B[t, T, j0, j2] = sum_j1 |Phi|^2 g3 / sinh(f1/2kT)   (q2 is the partner, swapped)
        in the same units as Gamma.
        """
        s = self.s
        dev = self.device
        triplets, weights = triplets_at_gp(s, gp)
        tr = torch.as_tensor(triplets, device=dev)
        wt = torch.as_tensor(weights, device=dev, dtype=torch.float64)
        f0 = self.freqs[gp]
        nT = len(self.temps)
        ise = torch.zeros((nT, self.nb), device=dev, dtype=torch.float64)
        colA, colB = [], []
        B = self._triplet_chunk()
        for i in range(0, len(triplets), B):
            t = tr[i:i + B]
            w = wt[i:i + B]
            qv = self.qbz[t]
            f = self.freqs[t]  # (B, 3, nb)
            e = self.eigvecs[t]
            pp = self.fn(qv, f, e)
            if self.sigma is None:
                f1v = self.freqs[self.nb_plus[t[:, 1]]].permute(0, 3, 1, 2)  # (B, nb, 24, 4)
                f2v = self.freqs[self.nb_minus[t[:, 2]]].permute(0, 3, 1, 2)
                gs = K.triplet_integration_weights(f0, f1v, f2v, max_elems=self.thm_elems,
                                                   vertex_limit=self.thm_vertex_limit,
                                                   with_g3=collision)
            else:
                gs = K.triplet_integration_weights_sigma(f0, f[:, 1], f[:, 2], self.sigma,
                                                         with_g3=collision)
            if self.average_degenerate_weights:
                Dg = self.degenerate_average_matrices()
                A1, A2 = Dg[t[:, 1]], Dg[t[:, 2]]
                gs = [torch.einsum("bjJ,biJK,bkK->bijk", A1, g, A2) for g in gs]
            g1, g2 = gs[0], gs[1]
            if collision:
                P3 = gs[2] * pp
                s1 = K.inv_sinh_occupation(f[:, 1], self.temps, s.cutoff_frequency)  # (nT, B, nb)
                s2 = K.inv_sinh_occupation(f[:, 2], self.temps, s.cutoff_frequency)
                colA.append(torch.einsum("bijk,tbk->btij", P3, s2))
                colB.append(torch.einsum("bijk,tbj->btik", P3, s1))
                del P3
            del gs
            # pp is exactly zero wherever f1 or f2 <= cutoff (where C sets n = -1 and
            # skips), so the occupation factors can be applied after band sums:
            # sum_{jk} [(n1_j + n2_k + 1) g1 + (n1_j - n2_k) g2] pp
            n1 = K.occupations(f[:, 1], self.temps, s.cutoff_frequency)  # (nT, B, nb)
            n2 = K.occupations(f[:, 2], self.temps, s.cutoff_frequency)
            P1 = g1 * pp
            P2 = g2 * pp
            del pp, g1, g2
            S1j, S1k = P1.sum(dim=3), P1.sum(dim=2)  # (B, nb0, nb)
            S2j, S2k = P2.sum(dim=3), P2.sum(dim=2)
            del P1, P2
            ise += torch.einsum("b,tbj,bij->ti", w, n1 + 1, S1j)
            ise += torch.einsum("b,tbk,bik->ti", w, n2, S1k)
            ise += torch.einsum("b,tbj,bij->ti", w, n1, S2j)
            ise -= torch.einsum("b,tbk,bik->ti", w, n2, S2k)
        unit = s.pp_unit_conversion * s.ise_unit_conversion
        ise *= unit
        if collision:
            return ise, torch.cat(colA) * unit, torch.cat(colB) * unit
        return ise

    # ------------------------------------------------------------------
    def _isotope_setup(self):
        s = self.s
        from phono3py.other.isotope import get_mass_variances
        mv = get_mass_variances(s.ph3.primitive)
        s.mass_variances = mv
        self.mv = torch.as_tensor(mv, device=self.device, dtype=torch.float64)
        g = torch.as_tensor(np.array(s.bz_grid.grg2bzg, dtype="int64"), device=self.device)
        self.iso_g = g
        fv = self.freqs[self.nb_plus[g]].permute(0, 3, 1, 2)  # (G, nb, 24, 4)
        self.iso_vs, self.iso_ci = K.sort_omegas(fv)

    def isotope_at_gp(self, gp):
        s = self.s
        f0 = self.freqs[gp]
        e0 = self.eigvecs[gp]
        G = len(self.iso_g)
        nb = self.nb
        num_grid = float(np.prod(s.bz_grid.D_diag))
        iw = K.thm_weights_pruned(f0, self.iso_vs.reshape(-1, 24, 4),
                                  self.iso_ci.reshape(-1, 24),
                                  max_elems=self.thm_elems,
                                  vertex_limit=self.thm_vertex_limit)  # (nb0, G*nb)
        iw = iw.reshape(nb, G, nb).permute(1, 0, 2)  # (G, nb0, nb)
        if self.average_degenerate_weights:
            Dg = self.degenerate_average_matrices()
            iw = torch.einsum("gjJ,giJ->gij", Dg[self.iso_g], iw)  # sets at q'
            iw = torch.einsum("iI,gIj->gij", Dg[gp], iw)  # sets at the mode of interest
        gamma = torch.zeros(nb, device=self.device, dtype=torch.float64)
        n_p = len(s.masses)
        step = max(1, int(self.mem_budget // (nb * nb * n_p * 16 * 4)))
        for i in range(0, G, step):
            gg = self.iso_g[i:i + step]
            gamma += K.isotope_gamma(f0, e0, self.freqs[gg], self.eigvecs[gg],
                                     iw[i:i + step], self.mv, s.cutoff_frequency, num_grid)
        if self.average_degenerate_weights:
            gamma = self.degenerate_average_matrices()[gp] @ gamma
        return gamma

    # ------------------------------------------------------------------
    def velocities(self, gps):
        """Velocity operators (n, nb, nb, 3) and group velocities (n, nb, 3)."""
        s = self.s
        q = self.qbz[torch.as_tensor(gps, device=self.device)]
        vops = []
        step = max(1, int(self.mem_budget // (self.dm.n_p * self.dm.n_s * 9 * 16 * 4)))
        for i in range(0, len(q), step):
            vops.append(K.velocity_operators(self.dm, q[i:i + step], s.lattice, s.factor,
                                             dynmat0=self.dm0))
        vop = torch.cat(vops)
        # group velocities: diagonal, and eigenvalues of the velocity operator inside
        # degenerate subspaces (tiny blocks -> host)
        vop_h = vop.cpu().numpy()
        gv = np.einsum("niix->nix", vop_h).real.copy()  # (n, nb, 3)
        freqs = self.freqs[torch.as_tensor(gps, device=self.device)].cpu().numpy()
        for n in range(len(gps)):
            pos = 0
            for deg in degenerate_sets(freqs[n]):
                if len(deg) > 1:
                    blk = vop_h[n, pos:pos + len(deg), pos:pos + len(deg), :]
                    for d in range(3):
                        gv[n, pos:pos + len(deg), d] = np.linalg.eigvalsh(blk[..., d])
                pos += len(deg)
        gv = torch.as_tensor(gv, device=self.device)
        return vop, gv

    def gv_by_gv_operator(self, vop, gps):
        """Sum over k* of v (x) v^* (n, nb, nb, 6) as in ConductivityWignerMixIn."""
        s = self.s
        R = torch.as_tensor(s.rotations_cartesian, device=self.device, dtype=torch.complex128)
        RR = torch.einsum("ria,rjb->ijab", R, R)  # sum_r R_ia R_jb
        out = torch.einsum("ijab,nsta,nstb->nstij", RR, vop, vop.conj())
        n_sum = 0
        for n, gp in enumerate(gps):
            ratio, order_kstar = kstar_info(s, gp)
            out[n] /= ratio
            n_sum += order_kstar
        idx = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        sum2 = torch.stack([out[..., a, b] for a, b in idx], dim=-1)
        return sum2, n_sum

    # ------------------------------------------------------------------
    @staticmethod
    def average_by_degeneracy(vals, freqs):
        """vals (..., nb) averaged over degenerate bands of freqs (nb,)."""
        out = vals.clone()
        for dset in degenerate_sets(freqs):
            if len(dset) > 1:
                out[..., dset] = vals[..., dset].mean(dim=-1, keepdim=True)
        return out

    def heat_capacity(self, f):
        """mode_cv (nT, nb) in eV/K as HeatCapacityMixIn._set_cv."""
        temps = self.temps[:, None]
        fe = f[None, :] * U.THzToEv
        cutoff = self.s.cutoff_frequency * U.THzToEv
        cond = (fe < 100 * temps * U.Kb) & (fe > cutoff)
        tt = torch.where(cond, temps.expand_as(cond), torch.full_like(fe, 10000.0))
        x = fe / (U.Kb * tt)
        ex = torch.exp(x)
        cv = U.Kb * x ** 2 * ex / (ex - 1.0) ** 2
        return torch.where(cond, cv, torch.zeros_like(cv))

    # ------------------------------------------------------------------
    def _run_gp_part(self, gp_part, is_isotope, progress=True):
        k, m = int(gp_part[0]), int(gp_part[1])
        gps = self.s.ir_grid_points
        sel = [i for i in range(len(gps)) if i % m == k]
        if is_isotope:
            self._isotope_setup()
        g_out = np.zeros((len(self.temps), len(sel), self.nb))
        gi_out = np.zeros((len(sel), self.nb))
        t1 = time.time()
        for j, i in enumerate(sel):
            gp = gps[i]
            g = self.gamma_at_gp(gp)
            g_out[:, j] = self.average_by_degeneracy(g, self.freqs[gp].cpu().numpy()).cpu().numpy()
            if is_isotope:
                gi_out[j] = self.isotope_at_gp(gp).cpu().numpy()
            if progress:
                el = time.time() - t1
                log(f"[GPU] part {k}/{m}: gp {j + 1}/{len(sel)} (index {i}, {el:.1f} s, "
                    f"eta {el / (j + 1) * (len(sel) - j - 1):.1f} s)")
        self.t_gamma = time.time() - t1
        return dict(partial=True, part=(k, m), indices=np.array(sel), grid_points=np.asarray(gps)[sel],
                    gamma=g_out, gamma_iso=gi_out, temperature=np.asarray(self.s.temperatures),
                    n_ir=len(gps), is_isotope=bool(is_isotope), t_gamma=self.t_gamma)

    def _load_gp_parts(self, files, is_isotope):
        gps = np.asarray(self.s.ir_grid_points)
        nT, n_ir = len(self.temps), len(gps)
        gamma = np.full((nT, n_ir, self.nb), np.nan)
        gamma_iso = np.zeros((n_ir, self.nb))
        m_seen = set()
        for f in files:
            q = np.load(f)
            if int(q["n_ir"]) != n_ir or not np.array_equal(q["grid_points"], gps[q["indices"]]):
                raise ValueError(f"{f}: grid points do not match this run")
            if not np.allclose(q["temperature"], self.s.temperatures):
                raise ValueError(f"{f}: temperatures do not match")
            if bool(q["is_isotope"]) != bool(is_isotope):
                raise ValueError(f"{f}: isotope setting does not match")
            gamma[:, q["indices"]] = q["gamma"]; gamma_iso[q["indices"]] = q["gamma_iso"]
            m_seen.add(int(q["part"][1]))
        if np.isnan(gamma).any():
            miss = np.where(np.isnan(gamma[0, :, 0]))[0]
            raise ValueError(f"missing grid points {miss.tolist()} (parts read: {len(files)}, m={sorted(m_seen)})")
        log(f"[GPU] linewidths of {n_ir} grid points taken from {len(files)} parts")
        t = lambda a: torch.as_tensor(a, device=self.device, dtype=torch.float64)
        return t(gamma), t(gamma_iso)

    def run(self, is_isotope=True, boundary_mfp=None, progress=True, lbte=False,
            pinv_cutoff=1e-8, pinv_method=0, gp_part=None, gamma_parts=None):
        """gp_part=(k, m): compute the linewidths only at irreducible points i with i % m == k and return them
        (no velocities, no kappa); gamma_parts=[files]: take all linewidths from such parts and compute kappa.
        The RTA linewidth of each grid point is independent of the others, so the split does not change the result.
        Not available with LBTE."""
        s = self.s
        dev = self.device
        gps = s.ir_grid_points
        n_ir = len(gps)
        nT = len(self.temps)
        nb = self.nb
        if (gp_part is not None or gamma_parts is not None) and lbte:
            raise ValueError("grid-point splitting is implemented for RTA only")
        if gp_part is not None:
            return self._run_gp_part(gp_part, is_isotope, progress)
        t0 = time.time()
        vop, gv = self.velocities(gps)
        sum2, N = self.gv_by_gv_operator(vop, gps)
        t_vel = time.time() - t0

        if is_isotope:
            self._isotope_setup()
        gamma = torch.zeros((nT, n_ir, nb), device=dev, dtype=torch.float64)
        gamma_iso = torch.zeros((n_ir, nb), device=dev, dtype=torch.float64)
        colmat = None
        if lbte:
            from .lbte import CollisionMatrix
            colmat = CollisionMatrix(self)
        t1 = time.time()
        if gamma_parts is not None:
            gamma, gamma_iso = self._load_gp_parts(gamma_parts, is_isotope)
        for i, gp in enumerate(gps if gamma_parts is None else []):
            if lbte:
                g, cA, cB = self.gamma_at_gp(gp, collision=True)
                colmat.add_row(i, gp, cA, cB)
                del cA, cB
            else:
                g = self.gamma_at_gp(gp)
            gamma[:, i] = self.average_by_degeneracy(g, self.freqs[gp].cpu().numpy())
            if is_isotope:
                gamma_iso[i] = self.isotope_at_gp(gp)
            if progress and (i % max(1, n_ir // 10) == 0 or i == n_ir - 1):
                el = time.time() - t1
                log(f"[GPU] gp {i + 1}/{n_ir} ({el:.1f} s, eta {el / (i + 1) * (n_ir - i - 1):.1f} s)")
        self.t_gamma = time.time() - t1

        # ---- Wigner RTA kappa (ConductivityWignerRTA.set_kappa_at_sigmas)
        freqs = self.freqs[torch.as_tensor(gps, device=dev)]  # (n_ir, nb)
        cutoff = s.cutoff_frequency
        conv = (U.THz * U.Angstrom) ** 2 * U.EV * U.Hbar / (s.volume * U.Angstrom ** 3)
        g_sum = gamma.clone()
        if is_isotope:
            g_sum = g_sum + gamma_iso[None]
        g_b = None
        if boundary_mfp is not None:
            g_b = torch.linalg.norm(gv, dim=-1) * U.Angstrom * 1e6 / (4 * np.pi * boundary_mfp)
            g_sum = g_sum + g_b[None]
        cv = torch.stack([self.heat_capacity(f) for f in freqs], dim=1)  # (nT, n_ir, nb)
        hw = freqs * U.THzToEv  # (n_ir, nb)
        ok = freqs > cutoff
        hw_s = torch.where(ok, hw, torch.ones_like(hw))
        hg = 2.0 * g_sum * U.THzToEv  # (nT, n_ir, nb)
        hg12 = hg[:, :, :, None] + hg[:, :, None, :]
        dw = hw[:, :, None] - hw[:, None, :]
        lor = 0.5 * hg12 / (dw[None] ** 2 + 0.25 * hg12 ** 2)
        pref = 0.25 * (hw[:, :, None] + hw[:, None, :])[None] * (
            cv[:, :, :, None] / hw_s[None, :, :, None] + cv[:, :, None, :] / hw_s[None, :, None, :])
        okk = (ok[:, :, None] & ok[:, None, :])[None]
        fac = torch.where(okk, pref * lor * conv, torch.zeros_like(lor))  # (nT, n_ir, nb, nb)
        contrib = fac[..., None] * sum2.real[None]  # (nT, n_ir, nb, nb, 6)
        deg = (torch.abs(freqs[:, :, None] - freqs[:, None, :]) < 1e-4)[None, :, :, :, None]
        cP = torch.where(deg, contrib, torch.zeros_like(contrib))
        cC = torch.where(deg, torch.zeros_like(contrib), contrib)
        mode_kP = 0.5 * cP.sum(dim=3) + 0.5 * cP.sum(dim=2)  # (nT, n_ir, nb, 6)
        kP = mode_kP.sum(dim=(1, 2)) / N
        kC = cC.sum(dim=(1, 2, 3)) / N
        self.result = dict(
            gamma=gamma.cpu().numpy()[None],
            gamma_iso=gamma_iso.cpu().numpy(),
            kappa_P_RTA=kP.cpu().numpy()[None],
            kappa_C=kC.cpu().numpy()[None],
            kappa=(kP + kC).cpu().numpy()[None],
            frequency=freqs.cpu().numpy(),
            group_velocity=gv.cpu().numpy(),
            heat_capacity=cv.cpu().numpy(),  # mode C_v (nT, n_ir, nb), eV/K, as phono3py
            qpoint=s.qpoints_bz[gps],
            weight=s.grid_weights,
            temperature=s.temperatures,
            mesh=np.array(s.bz_grid.D_diag, dtype="int64"),
            num_sampling_grid_points=N,
            t_velocity=t_vel,
        )
        if lbte:
            t2 = time.time()
            kL, kR = colmat.solve(gamma, gamma_iso if is_isotope else None, gv,
                                  pinv_cutoff=pinv_cutoff, pinv_method=pinv_method, boundary=g_b)
            self.t_lbte_solve = time.time() - t2
            self.collision_eigenvalues = [e.cpu().numpy() for e in colmat.eigenvalues]
            self.result.update(
                kappa_P_exact=kL.cpu().numpy()[None],
                kappa_P_RTA_lbte=kR.cpu().numpy()[None],
                kappa_TOT_exact=(kL + kC).cpu().numpy()[None],
                t_lbte_solve=self.t_lbte_solve,
            )
        return self.result
