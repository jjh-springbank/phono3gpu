# -*- coding: utf-8 -*-
"""Atom-resolved phonon scattering (ph-ph vertex split by atom) on top of PF-GPU.

Analysis only: imports the PF-GPU kernels / GPUConductivity read-only and never
changes the main kappa path.

Vertex split
------------
Phi(l0, l1, l2) = sum_i Phi_i, Phi_i = sum_{a, j, k} F[i a, j, k] E0[i a, l0] E1[j, l1] E2[k, l2]
(F = fc3 in reciprocal space after the r0 average, E = e / sqrt(m); i = atom of the
mode of interest).  The atom-resolved linewidth uses Re(Phi_i^* Phi) in place of |Phi|^2:

    gamma_{l0, i} = unit * sum_t w_t sum_{l1 l2} Re(Phi_i^* Phi) / (f0 f1 f2) * W_T(l0, l1, l2)
    W_T = (n1 + n2 + 1) g1 + (n1 - n2) g2

with exactly the integration weights g1/g2 of GPUConductivity.gamma_at_gp, so
sum_i gamma_{l0, i} = gamma_{l0} (sum rule).  Isotope scattering is a sum over atoms
already (mass variance of atom i times |e_i . e'_i|^2), split the same way.

Per-atom attribution of kappa
-----------------------------
s_{lam, i} = (gamma_{lam, i} + gamma_iso_{lam, i}) / (gamma_lam + gamma_iso_lam)
S_P_i = sum_lam kappa_P_lam s_{lam, i}  (isotropic trace/3),  sum_i S_P_i = kappa_P
S_C_i = sum_{lam, lam'} kappa_C(lam, lam') (s_{lam, i} + s_{lam', i}) / 2,  sum_i S_C_i = kappa_C
S_i is the first-order kappa loss if the scattering located at atom i were doubled
(-sum_lam gamma_{lam, i} d kappa / d gamma_lam for the RTA part).
Per-atom sums over the irreducible q-points are averaged over symmetry-equivalent
atoms (orbit mean) = the full-mesh sum.
"""
import time

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import kstar_info, log, triplets_at_gp


class AtomScatter:
    """Atom-resolved scattering on a ready GPUConductivity object ``c``."""

    CHANNELS = [f"{p}_{u}_{k}" for p in ("decay", "coalescence") for u in ("N", "U") for k in ("LL", "LH", "HH")]
    ATOM_CHANNELS = [f"{p}_{k}" for p in ("decay", "coalescence") for k in ("LL", "LH", "HH")]
    ATOM_CHANNELS_OLD = ["decay_LL", "decay_other", "coalescence_LL", "coalescence_other"]  # parts written before 2026-11-22 18h

    def __init__(self, c, fcut=None):
        """fcut (THz): if set, also split the ph-ph linewidth into scattering channels. Decay = the mode splits
        into two (weight (n1+n2+1) delta(w-w1-w2)), coalescence = it merges with a second phonon (weight
        (n1-n2) delta terms); N/U = normal/umklapp triplet (q+q1+q2 = 0 or G); LL/LH/HH = both / one / none of
        the two partner phonons below fcut. Atom-resolved channels: decay or coalescence x (LL or other)."""
        self.fcut = fcut
        self.c = c
        self.s = c.s
        self.dev = c.device
        self.n_p = len(self.s.masses)
        self.nb = c.nb
        self.nT = len(c.temps)
        self.map_atoms = np.array(self.s.ph3.primitive_symmetry.get_map_atoms(), dtype="int64")
        sym = self.s.ph3.primitive_symmetry
        self._rinvT = np.array([np.linalg.inv(r.T) for r in sym.dataset.rotations])
        self._perms = np.array(sym.atomic_permutations, dtype="int64")

    def little_group_perms(self, gp):
        """Atom permutations of the space-group ops that map q to +-q (mod G).

        triplets_at_gp reduces the triplets with the little group of q and keeps one
        representative per class with its weight; the total gamma is invariant under
        that, the atom split is not (the ops permute atoms). Averaging the atom axis
        over the little group restores the full triplet sum."""
        q = self.s.qpoints_bz[gp]
        qr = self._rinvT @ q
        ok = [np.allclose(d, np.round(d), atol=1e-8) for d in (qr - q)]
        ok = np.array(ok) | np.array([np.allclose(d, np.round(d), atol=1e-8) for d in (qr + q)])
        return self._perms[ok]

    def little_group_average(self, a, gp):
        """a (..., n_p) -> mean over the little group of a[..., perm]."""
        P = self.little_group_perms(gp)
        return a[..., P].mean(axis=-2) if isinstance(a, np.ndarray) else a[..., torch.as_tensor(P, device=a.device)].mean(dim=-2)

    # ------------------------------------------------------------------
    def _triplet_chunk(self):
        c, nb = self.c, self.nb
        if c.chunk_qp is not None:
            return int(c.chunk_qp)
        csize = 8 if c.cdtype == torch.complex64 else 16
        # F + A1..A3 + permute temps, X1, Y, X, Q_T, Z  (complex)  + inv_fff, g1, g2, W_T, pp, P (real)
        per = 10 * nb ** 3 * csize + 8 * nb ** 3 * 8
        if c.sigma is None:
            from ..conductivity import _bytes_per_triplet_thm
            per = max(per, _bytes_per_triplet_thm(nb))
        return max(1, int(c.mem_budget // per))

    def _l_block(self, B):
        """Number of modes of q0 handled at once: the per-mode work arrays (Phi, |Phi|^2, 1/www, g1, g2, weights and
        temporaries) take about 200 nb^2 bytes per mode and triplet; give them a quarter of the budget."""
        nb = self.nb
        return max(1, min(nb, int(0.25 * self.c.mem_budget // (max(1, B) * nb * nb * 200))))

    def gamma_atom_at_gp(self, gp):
        """(gamma_atom (nT, nb, n_p), gamma_total (nT, nb)) at BZ grid point gp, ph-ph only.

        gamma_total is computed from |Phi|^2 in the same pass (independent of the split)."""
        c, s, dev = self.c, self.s, self.dev
        nb, n_p, nT = self.nb, self.n_p, self.nT
        fn = c.fn
        triplets, weights = triplets_at_gp(s, gp)
        tr = torch.as_tensor(triplets, device=dev)
        wt = torch.as_tensor(weights, device=dev, dtype=torch.float64)
        f0 = c.freqs[gp]
        ga = torch.zeros((nT, nb, n_p), device=dev, dtype=torch.float64)
        gt = torch.zeros((nT, nb), device=dev, dtype=torch.float64)
        chan = self.fcut is not None
        if chan:
            gch = torch.zeros((nT, nb, 12), device=dev, dtype=torch.float64)
            gach = torch.zeros((nT, nb, n_p, 6), device=dev, dtype=torch.float64)
        B = self._triplet_chunk()
        for i in range(0, len(triplets), B):
            t = tr[i:i + B]
            w = wt[i:i + B]
            Bc = len(t)
            qv = c.qbz[t]
            f = c.freqs[t]  # (B, 3, nb)
            e = c.eigvecs[t]
            # --- vertex with the atom index of the mode of interest kept open (Fc3Normal.__call__)
            E = (e * fn.inv_sqrt_m[None, None, :, None]).to(fn.cdtype)
            # Y (B, nb_cart0, nb1, nb2) = F contracted with the two partner modes. The route through the full F is
            # ~30% faster but holds ~8 arrays of nb^3; large cells use the term-by-term contraction (~4 arrays).
            csz = 8 if fn.cdtype == torch.complex64 else 16
            if 8 * Bc * nb ** 3 * csz > 0.8 * c.mem_budget:
                Y = fn.contract_partners(qv, E)
            else:
                F = fn.fc3_reciprocal(qv)
                X1 = torch.einsum("bijk,bkc->bijc", F, E[:, 2])
                del F
                Y = torch.einsum("bijc,bjm->bimc", X1, E[:, 1])
                del X1
            # --- everything below is independent for each mode l of q0: process l in blocks to bound memory
            ok = f > fn.cutoff
            f_safe = torch.where(ok, f, torch.ones_like(f))
            if c.sigma is None:
                f1v = c.freqs[c.nb_plus[t[:, 1]]].permute(0, 3, 1, 2)
                f2v = c.freqs[c.nb_minus[t[:, 2]]].permute(0, 3, 1, 2)
                f1c = torch.clamp(f1v, min=0)[:, :, None]
                f2c = torch.clamp(f2v, min=0)[:, None, :]
                vs, ci = K.sort_omegas(torch.stack((f2c - f1c, f1c - f2c, f1c + f2c), dim=1))  # (B, 3, nb, nb, 24, 4)
                vs, ci = vs.reshape(-1, 24, 4), ci.reshape(-1, 24)
                del f1v, f2v, f1c, f2c
            if c.average_degenerate_weights:
                Dg = c.degenerate_average_matrices()
                A1, A2 = Dg[t[:, 1]], Dg[t[:, 2]]
            n1 = K.occupations(f[:, 1], c.temps, s.cutoff_frequency)  # (nT, B, nb)
            n2 = K.occupations(f[:, 2], c.temps, s.cutoff_frequency)
            E0c_all = E[:, 0].conj().reshape(Bc, n_p, 3, nb)  # [b, i, a, l]
            if not chan:
                Yh = Y.reshape(Bc, nb, nb * nb).conj().transpose(1, 2)  # (B, nb1*nb2, nb_cart0)
            if chan:
                umk = (torch.round(qv.sum(dim=1)).abs().sum(dim=-1) > 0.5)  # (B,) q+q1+q2 = G != 0
                lo1, lo2 = f[:, 1] < self.fcut, f[:, 2] < self.fcut  # (B, nb)
                LL = (lo1[:, :, None] & lo2[:, None, :])[:, None]  # (B, 1, nb1, nb2)
                HH = (~lo1[:, :, None] & ~lo2[:, None, :])[:, None]
                cls = [LL, ~(LL | HH), HH]
                wN, wU = w * (~umk), w * umk
            lb = self._l_block(Bc)
            for l0 in range(0, nb, lb):
                ls = slice(l0, min(l0 + lb, nb))
                X = torch.einsum("bimc,bil->blmc", Y, E[:, 0][:, :, ls])  # Phi (B, nl, nb1, nb2)
                fff = f_safe[:, 0, ls, None, None] * f_safe[:, 1, None, :, None] * f_safe[:, 2, None, None, :]
                okm = ok[:, 0, ls, None, None] & ok[:, 1, None, :, None] & ok[:, 2, None, None, :]
                inv = torch.where(okm, 1.0 / fff, torch.zeros_like(fff))
                del fff, okm
                pp = (X.real.to(torch.float64) ** 2 + X.imag.to(torch.float64) ** 2) * inv
                # --- integration weights: same calls as GPUConductivity.gamma_at_gp, for the modes of this block
                if c.sigma is None:
                    wgt = K.thm_weights_pruned(f0[ls], vs, ci, max_elems=c.thm_elems, vertex_limit=c.thm_vertex_limit)
                    wgt = wgt.reshape(ls.stop - ls.start, Bc, 3, nb, nb).permute(1, 2, 0, 3, 4)
                    gs = (wgt[:, 2], wgt[:, 0] - wgt[:, 1])
                    del wgt
                else:
                    gs = K.triplet_integration_weights_sigma(f0[ls], f[:, 1], f[:, 2], c.sigma)
                if c.average_degenerate_weights:
                    gs = [torch.einsum("bjJ,biJK,bkK->bijk", A1, g, A2) for g in gs]
                g1, g2 = gs
                del gs
                E0c = E0c_all[..., ls]
                for it in range(nT):
                    Wd = g1 * (n1[it][:, None, :, None] + n2[it][:, None, None, :] + 1)
                    Wc = g2 * (n1[it][:, None, :, None] - n2[it][:, None, None, :])
                    Wt = Wd + Wc
                    gt[it, ls] += torch.einsum("b,bijk->i", w, pp * Wt)
                    if not chan:
                        Q = (X * (inv * Wt).to(X.dtype)).reshape(Bc, ls.stop - ls.start, nb * nb)
                        Z = torch.matmul(Q, Yh).reshape(Bc, ls.stop - ls.start, n_p, 3)  # [b, l, i, a]
                        del Q
                        ga[it, ls] += torch.einsum("b,bial,blia->li", w, E0c, Z).real.to(torch.float64)
                        del Z
                    if chan:
                        for ip, Wp in enumerate((Wd, Wc)):
                            P = pp * Wp
                            for iu, wu in enumerate((wN, wU)):
                                for ic, m in enumerate(cls):
                                    gch[it, ls, ip * 6 + iu * 3 + ic] += torch.einsum("b,bijk->i", wu, torch.where(m, P, 0.0))
                        del P
                    del Wt, Wd, Wc
                if chan:
                    self._atom_channels_factored(gach[:, ls], X, Y, inv, g1, g2, n1, n2, E0c, lo1, lo2, w)
                del X, pp, inv, g1, g2, E0c
            del Y
        unit = s.pp_unit_conversion * s.ise_unit_conversion
        if chan:
            return gach.sum(-1) * unit, gt * unit, gch * unit, gach * unit
        return ga * unit, gt * unit

    def _atom_channels_factored(self, gach, X, Y, inv, g1, g2, n1, n2, E0c, lo1, lo2, w):
        """Atom-resolved linewidth per channel, with 4 contractions of order nb^4 per triplet instead of one
        (nb x nb^2)(nb^2 x nb) product per temperature and channel.

        gamma_i[l] = Re sum_a E0*[ia,l] sum_mn Q0[l,m,n] W_T[m,n] Y*[ia,m,n],  Q0 = Phi * inv * g.
        The temperature enters only through n1[m] and n2[n] (decay: g1 (n1 + n2 + 1), coalescence: g2 (n1 - n2)), and
        the partner classes through m and n separately. So contract n first, U[l,m,ia] = sum_n Q0 Y*, and m first,
        V[l,n,ia] = sum_m Q0 Y*, with the inner sum split into partners below / above fcut; temperatures and classes
        are then sums over the remaining index (order nb^3). Accumulates into gach (nT, nb, n_p, 6):
        [decay LL, LH, HH, coalescence LL, LH, HH]."""
        nb, n_p, nT = self.nb, self.n_p, self.nT
        nl = X.shape[1]  # modes of q0 in this block
        blk = max(1, int(1.0e9 // (nl * nb * 16 * 4)))
        for b in range(X.shape[0]):
            Xb, Yb, E0 = X[b], Y[b], E0c[b]  # (l, m, n), (ia, m, n), (i, a, l)
            # frequencies are ascending, so "below fcut" is a leading range: slices (views), no copies
            k1, k2 = int(lo1[b].sum()), int(lo2[b].sum())
            assert bool(lo1[b][:k1].all()) and bool(lo2[b][:k2].all()), "frequencies not sorted"
            r1 = [slice(0, k1), slice(k1, nb)]  # m below / above
            r2 = [slice(0, k2), slice(k2, nb)]
            for ip, g in enumerate((g1, g2)):
                sg = 1.0 if ip == 0 else -1.0
                for i0 in range(0, nb, blk):
                    bs = slice(i0, min(i0 + blk, nb))
                    c1 = 0 if bs.stop <= k1 else (1 if bs.start >= k1 else None)   # class of the kept index m
                    c2 = 0 if bs.stop <= k2 else (1 if bs.start >= k2 else None)   # class of the kept index n
                    # U: m in block (kept), n summed over class range; weights n1[m] (+1 for decay)
                    for cn, rn in enumerate(r2):
                        if rn.stop == rn.start:
                            continue
                        Q = (Xb[:, bs, rn] * (inv[b][:, bs, rn] * g[b][:, bs, rn]).to(Xb.dtype)).permute(1, 0, 2)  # (mb, l, n)
                        U = torch.bmm(Q, Yb[:, bs, rn].conj().permute(1, 2, 0))  # (mb, l, ia)
                        del Q
                        R = torch.einsum("ial,mlia->mli", E0, U.reshape(-1, nl, n_p, 3)).real.to(torch.float64)
                        del U
                        for it in range(nT):
                            cm = n1[it][b, bs] + (1.0 if ip == 0 else 0.0)
                            if c1 is not None:
                                gach[it, :, :, ip * 3 + c1 + cn] += w[b] * torch.einsum("m,mli->li", cm, R)
                            else:  # block straddles fcut
                                lo = lo1[b][bs].to(torch.float64)
                                gach[it, :, :, ip * 3 + 0 + cn] += w[b] * torch.einsum("m,mli->li", cm * lo, R)
                                gach[it, :, :, ip * 3 + 1 + cn] += w[b] * torch.einsum("m,mli->li", cm * (1 - lo), R)
                        del R
                    # V: n in block (kept), m summed over class range; weights +-n2[n]
                    for cm_, rm in enumerate(r1):
                        if rm.stop == rm.start:
                            continue
                        Q = (Xb[:, rm, bs] * (inv[b][:, rm, bs] * g[b][:, rm, bs]).to(Xb.dtype)).permute(2, 0, 1)  # (nb_, l, m)
                        V = torch.bmm(Q, Yb[:, rm, bs].conj().permute(2, 1, 0))  # (nb_, l, ia)
                        del Q
                        R = torch.einsum("ial,nlia->nli", E0, V.reshape(-1, nl, n_p, 3)).real.to(torch.float64)
                        del V
                        for it in range(nT):
                            cn_ = sg * n2[it][b, bs]
                            if c2 is not None:
                                gach[it, :, :, ip * 3 + cm_ + c2] += w[b] * torch.einsum("n,nli->li", cn_, R)
                            else:
                                lo = lo2[b][bs].to(torch.float64)
                                gach[it, :, :, ip * 3 + cm_ + 0] += w[b] * torch.einsum("n,nli->li", cn_ * lo, R)
                                gach[it, :, :, ip * 3 + cm_ + 1] += w[b] * torch.einsum("n,nli->li", cn_ * (1 - lo), R)
                        del R

    # ------------------------------------------------------------------
    def isotope_atom_at_gp(self, gp):
        """gamma_iso split by atom (nb, n_p); mirrors GPUConductivity.isotope_at_gp."""
        c, s = self.c, self.s
        if not hasattr(c, "iso_g"):
            c._isotope_setup()
        f0 = c.freqs[gp]
        e0 = c.eigvecs[gp]
        G = len(c.iso_g)
        nb, n_p = self.nb, self.n_p
        num_grid = float(np.prod(s.bz_grid.D_diag))
        iw = K.thm_weights_pruned(f0, c.iso_vs.reshape(-1, 24, 4), c.iso_ci.reshape(-1, 24),
                                  max_elems=c.thm_elems, vertex_limit=c.thm_vertex_limit)
        iw = iw.reshape(nb, G, nb).permute(1, 0, 2)
        if c.average_degenerate_weights:
            Dg = c.degenerate_average_matrices()
            iw = torch.einsum("gjJ,giJ->gij", Dg[c.iso_g], iw)
            iw = torch.einsum("iI,gIj->gij", Dg[gp], iw)
        gamma = torch.zeros((nb, n_p), device=self.dev, dtype=torch.float64)
        step = max(1, int(c.mem_budget // (nb * nb * n_p * 16 * 6)))
        e0r = e0.reshape(n_p, 3, -1).conj()
        for i in range(0, G, step):
            gg = c.iso_g[i:i + step]
            fg, eg = c.freqs[gg], c.eigvecs[gg]
            ov = torch.einsum("lmj,glmk->gjkl", e0r, eg.reshape(len(gg), n_p, 3, nb))
            sv = (ov.real ** 2 + ov.imag ** 2) * c.mv  # (g, nb0, nb, n_p)
            okk = (fg >= s.cutoff_frequency)[:, None, :, None]
            sv = torch.where(okk, sv * iw[i:i + step][..., None], torch.zeros_like(sv))
            gamma += sv.sum(dim=(0, 2))
        gamma = gamma * (K.M_2PI / 4 * f0 * f0 / 2)[:, None]
        gamma = torch.where((f0 < s.cutoff_frequency)[:, None], torch.zeros_like(gamma), gamma)
        gamma = gamma / num_grid
        if c.average_degenerate_weights:
            gamma = c.degenerate_average_matrices()[gp] @ gamma
        return gamma

    # ------------------------------------------------------------------
    def local_dos(self, sigma=0.25, f_low=4.0, df=0.05):
        """Atom-projected DOS on the regular grid (all q, no symmetry needed).

        Returns pdos (n_p, nw) [states/THz per atom, sum over atoms and w*df = 3 n_p],
        freq points, and low-frequency weight sum_{q, 0<f<f_low} |e_i|^2 / N per atom."""
        c, s = self.c, self.s
        g = torch.as_tensor(np.array(s.bz_grid.grg2bzg, dtype="int64"), device=self.dev)
        fr = c.freqs[g]  # (N, nb)
        ev = c.eigvecs[g]  # (N, nb, nb) columns = modes
        N = len(g)
        p = (ev.abs() ** 2).reshape(N, self.n_p, 3, self.nb).sum(2)  # (N, n_p, nb)
        fmax = float(fr.max()) + 4 * sigma
        w = torch.arange(0.0, fmax + df, df, device=self.dev, dtype=torch.float64)
        pdos = torch.zeros((self.n_p, len(w)), device=self.dev, dtype=torch.float64)
        for i in range(0, N, 16):
            gs = K.gaussian(w[:, None, None] - fr[None, i:i + 16], sigma)  # (nw, n, nb)
            pdos += torch.einsum("wnm,nam->aw", gs, p[i:i + 16])
        pdos /= N
        low = ((fr > s.cutoff_frequency) & (fr < f_low)).to(torch.float64)
        wlow = torch.einsum("nm,nam->a", low, p) / N
        return pdos.cpu().numpy(), w.cpu().numpy(), wlow.cpu().numpy()

    # ------------------------------------------------------------------
    def mode_kappa(self, gamma, gamma_iso, gv, sum2, N):
        """Copy of the Wigner-RTA post-processing of GPUConductivity.run, returning the
        mode-resolved pieces: mode_kP (nT, n_ir, nb, 6) and mode_kC (nT, n_ir, nb, 6) with
        mode_kC = half row + half column sums of the coherence matrix (sum = kappa_C)."""
        from phonopy import units as U
        c, s, dev = self.c, self.s, self.dev
        gps = s.ir_grid_points
        freqs = c.freqs[torch.as_tensor(gps, device=dev)]
        cutoff = s.cutoff_frequency
        conv = (U.THz * U.Angstrom) ** 2 * U.EV * U.Hbar / (s.volume * U.Angstrom ** 3)
        g_sum = gamma + gamma_iso[None]
        cv = torch.stack([c.heat_capacity(f) for f in freqs], dim=1)  # (nT, n_ir, nb)
        hw = freqs * U.THzToEv
        ok = freqs > cutoff
        hw_s = torch.where(ok, hw, torch.ones_like(hw))
        hg = 2.0 * g_sum * U.THzToEv
        hg12 = hg[:, :, :, None] + hg[:, :, None, :]
        dw = hw[:, :, None] - hw[:, None, :]
        lor = 0.5 * hg12 / (dw[None] ** 2 + 0.25 * hg12 ** 2)
        pref = 0.25 * (hw[:, :, None] + hw[:, None, :])[None] * (
            cv[:, :, :, None] / hw_s[None, :, :, None] + cv[:, :, None, :] / hw_s[None, :, None, :])
        okk = (ok[:, :, None] & ok[:, None, :])[None]
        fac = torch.where(okk, pref * lor * conv, torch.zeros_like(lor))
        contrib = fac[..., None] * sum2.real[None]
        deg = (torch.abs(freqs[:, :, None] - freqs[:, None, :]) < 1e-4)[None, :, :, :, None]
        cP = torch.where(deg, contrib, torch.zeros_like(contrib))
        cC = torch.where(deg, torch.zeros_like(contrib), contrib)
        del contrib
        mode_kP = (0.5 * cP.sum(dim=3) + 0.5 * cP.sum(dim=2)) / N
        mode_kC = (0.5 * cC.sum(dim=3) + 0.5 * cC.sum(dim=2)) / N
        return mode_kP, mode_kC, cv

    def orbit_mean(self, a):
        """Average the last (atom) axis over symmetry-equivalent atoms."""
        out = np.empty_like(a)
        for r in np.unique(self.map_atoms):
            m = self.map_atoms == r
            out[..., m] = a[..., m].mean(axis=-1, keepdims=True)
        return out

    # ------------------------------------------------------------------
    def _channels_at(self, gp, gch, gach, fq):
        """degeneracy average (band axis) of the channel arrays; little-group average of the atom channels;
        the 4th atom channel (coalescence_other) is the total minus the three computed ones (filled by caller)."""
        c = self.c
        gch = c.average_by_degeneracy(gch.transpose(1, 2), fq).transpose(1, 2)  # (nT, nb, 12)
        g = gach.permute(0, 3, 2, 1)  # (nT, 3, n_p, nb)
        g = c.average_by_degeneracy(g.reshape(-1, g.shape[2], g.shape[3]), fq).reshape(g.shape)
        g = g.permute(0, 3, 1, 2)  # (nT, nb, 3, n_p)
        g = self.little_group_average(g, gp)
        return gch, g.permute(0, 1, 3, 2)  # (nT, nb, n_p, 3)

    def _gp_part(self, gp_part, is_isotope, progress=True):
        """Linewidths (and their atom split) at irreducible points i with i % m == k only."""
        c, gps = self.c, self.s.ir_grid_points
        k, m = int(gp_part[0]), int(gp_part[1])
        sel = [i for i in range(len(gps)) if i % m == k]
        if is_isotope:
            c._isotope_setup()
        ga_o = np.zeros((self.nT, len(sel), self.nb, self.n_p)); g_o = np.zeros((self.nT, len(sel), self.nb))
        gi_o = np.zeros((len(sel), self.nb, self.n_p))
        if self.fcut is not None:
            gc_o = np.zeros((self.nT, len(sel), self.nb, 12)); gac_o = np.zeros((self.nT, len(sel), self.nb, self.n_p, 6))
        t1 = time.time()
        for j, i in enumerate(sel):
            gp = gps[i]
            out = self.gamma_atom_at_gp(gp)
            ga, gt = out[:2]
            fq = c.freqs[gp].cpu().numpy()
            if self.fcut is not None:
                gch, gach = self._channels_at(gp, out[2], out[3], fq)
                gc_o[:, j] = gch.cpu().numpy(); gac_o[:, j] = gach.cpu().numpy()
            g_o[:, j] = c.average_by_degeneracy(gt, fq).cpu().numpy()
            ga = c.average_by_degeneracy(ga.transpose(1, 2), fq).transpose(1, 2)
            ga_o[:, j] = self.little_group_average(ga, gp).cpu().numpy()
            if is_isotope:
                gi_o[j] = self.isotope_atom_at_gp(gp).cpu().numpy()
            if progress:
                el = time.time() - t1
                log(f"[ATOM] part {k}/{m}: gp {j + 1}/{len(sel)} (index {i}, {el:.1f} s, "
                    f"eta {el / (j + 1) * (len(sel) - j - 1):.1f} s)")
        d = dict(partial=True, part=(k, m), indices=np.array(sel), grid_points=np.asarray(gps)[sel], n_ir=len(gps),
                 gamma=g_o, gamma_atom=ga_o, gamma_isotope_atom=gi_o, temperature=np.asarray(self.s.temperatures),
                 is_isotope=bool(is_isotope), t_gamma=time.time() - t1)
        if self.fcut is not None:
            d.update(gamma_channel=gc_o, gamma_atom_channel6=gac_o, channel_fcut=self.fcut)
        return d

    def _load_parts(self, files, is_isotope):
        gps = np.asarray(self.s.ir_grid_points); n_ir = len(gps)
        g = np.full((self.nT, n_ir, self.nb), np.nan); ga = np.zeros((self.nT, n_ir, self.nb, self.n_p))
        gi = np.zeros((n_ir, self.nb, self.n_p))
        for f in files:
            q = np.load(f)
            if int(q["n_ir"]) != n_ir or not np.array_equal(q["grid_points"], gps[q["indices"]]):
                raise ValueError(f"{f}: grid points do not match this run")
            if not np.allclose(q["temperature"], self.s.temperatures) or bool(q["is_isotope"]) != bool(is_isotope):
                raise ValueError(f"{f}: temperatures or isotope setting do not match")
            if q["gamma_atom"].shape[-1] != self.n_p:
                raise ValueError(f"{f}: number of atoms does not match")
            ii = q["indices"]; g[:, ii] = q["gamma"]; ga[:, ii] = q["gamma_atom"]; gi[ii] = q["gamma_isotope_atom"]
        if np.isnan(g).any():
            raise ValueError(f"missing grid points {np.where(np.isnan(g[0, :, 0]))[0].tolist()}")
        log(f"[ATOM] linewidths of {n_ir} grid points taken from {len(files)} parts")
        t = lambda a: torch.as_tensor(a, device=self.dev, dtype=torch.float64)
        self._loaded_channels = None
        if self.fcut is not None:
            keys = {("gamma_atom_channel6" if "gamma_atom_channel6" in np.load(f) else "gamma_atom_channel3") for f in files}
            if len(keys) != 1:
                raise ValueError("parts mix old (3 computed atom channels) and new (6) channel formats")
            key = keys.pop()
            gc = np.zeros((self.nT, n_ir, self.nb, 12)); gac = np.zeros((self.nT, n_ir, self.nb, self.n_p, int(key[-1])))
            for f in files:
                q = np.load(f)
                if "gamma_channel" not in q or abs(float(q["channel_fcut"]) - self.fcut) > 1e-9:
                    raise ValueError(f"{f}: no channel data for fcut={self.fcut}")
                gc[:, q["indices"]] = q["gamma_channel"]; gac[:, q["indices"]] = q[key]
            self._loaded_channels = (t(gc), t(gac))
        return t(g), t(ga), t(gi)

    def run(self, is_isotope=True, progress=True, gp_part=None, parts=None):
        """gp_part=(k, m): only the grid points i % m == k, returned as a partial result (no kappa, no maps);
        parts=[files]: take all linewidths from such partial results. Grid points are independent in the RTA."""
        c, s, dev = self.c, self.s, self.dev
        gps = s.ir_grid_points
        n_ir, nT, nb, n_p = len(gps), self.nT, self.nb, self.n_p
        if gp_part is not None:
            return self._gp_part(gp_part, is_isotope, progress)
        t0 = time.time()
        # velocity operators in small q batches: one batched eigh over all 34 q of a 624-band cell failed in
        # cusolver (CUSOLVER_STATUS_INTERNAL_ERROR, Si/diamond 208 atoms, 2026-11-20); batch size does not change results
        mb = c.mem_budget
        c.mem_budget = min(mb, 4 * self.n_p * c.dm.n_s * 9 * 16 * 4)
        try:
            vop, gv = c.velocities(gps)
        finally:
            c.mem_budget = mb
        sum2, N = c.gv_by_gv_operator(vop, gps)
        del vop
        t_vel = time.time() - t0
        if is_isotope:
            c._isotope_setup()
        gamma_atom = torch.zeros((nT, n_ir, nb, n_p), device=dev, dtype=torch.float64)
        gamma = torch.zeros((nT, n_ir, nb), device=dev, dtype=torch.float64)
        giso_atom = torch.zeros((n_ir, nb, n_p), device=dev, dtype=torch.float64)
        t1 = time.time()
        if self.fcut is not None:
            gamma_ch = torch.zeros((nT, n_ir, nb, 12), device=dev, dtype=torch.float64)
            gatom_ch = torch.zeros((nT, n_ir, nb, n_p, 6), device=dev, dtype=torch.float64)
        if parts is not None:
            gamma, gamma_atom, giso_atom = self._load_parts(parts, is_isotope)
            if self.fcut is not None:
                gamma_ch, gatom_ch = self._loaded_channels
        for i, gp in enumerate(gps if parts is None else []):
            out = self.gamma_atom_at_gp(gp)
            ga, gt = out[:2]
            fq = c.freqs[gp].cpu().numpy()
            if self.fcut is not None:
                gamma_ch[:, i], gatom_ch[:, i] = self._channels_at(gp, out[2], out[3], fq)
            gamma[:, i] = c.average_by_degeneracy(gt, fq)
            ga = c.average_by_degeneracy(ga.transpose(1, 2), fq).transpose(1, 2)
            gamma_atom[:, i] = self.little_group_average(ga, gp)
            if is_isotope:
                giso_atom[i] = self.isotope_atom_at_gp(gp)  # full G sum: no little-group reduction
            if progress and (i % max(1, n_ir // 10) == 0 or i == n_ir - 1):
                el = time.time() - t1
                log(f"[ATOM] gp {i + 1}/{n_ir} ({el:.1f} s, eta {el / (i + 1) * (n_ir - i - 1):.1f} s)")
        t_gamma = time.time() - t1
        gamma_iso = giso_atom.sum(-1)
        mode_kP, mode_kC, cv = self.mode_kappa(gamma, gamma_iso, gv, sum2, N)

        # ---- per-atom attribution (host, small arrays)
        g_tot = (gamma + gamma_iso[None]).cpu().numpy()  # (nT, n_ir, nb)
        g_tot_atom = (gamma_atom + giso_atom[None]).cpu().numpy()  # (nT, n_ir, nb, n_p)
        safe = np.where(g_tot > 0, g_tot, 1.0)
        share = np.where((g_tot > 0)[..., None], g_tot_atom / safe[..., None], 0.0)
        kP_iso = mode_kP[..., :3].mean(-1).cpu().numpy()  # (nT, n_ir, nb)
        kC_iso = mode_kC[..., :3].mean(-1).cpu().numpy()
        SP_raw = np.einsum("tqm,tqma->ta", kP_iso, share)
        SC_raw = np.einsum("tqm,tqma->ta", kC_iso, share)
        # heat-carrier weighted scattering rate: sum C v^2 gamma_i / sum C v^2 (star-weighted)
        w = torch.as_tensor(s.grid_weights, device=dev, dtype=torch.float64)
        cv2 = (cv * (gv ** 2).sum(-1)[None] / 3 * w[None, :, None]).cpu().numpy()  # (nT, n_ir, nb)
        Gam_raw = np.einsum("tqm,tqma->ta", cv2, g_tot_atom) / cv2.sum(axis=(1, 2))[:, None]
        res = dict(
            temperature=s.temperatures, mesh=np.array(s.bz_grid.D_diag, dtype="int64"),
            qpoint=s.qpoints_bz[gps], weight=s.grid_weights,
            frequency=c.freqs[torch.as_tensor(gps, device=dev)].cpu().numpy(),
            group_velocity=gv.cpu().numpy(), heat_capacity=cv.cpu().numpy(),
            gamma=gamma.cpu().numpy(), gamma_atom=gamma_atom.cpu().numpy(),
            gamma_isotope=gamma_iso.cpu().numpy(), gamma_isotope_atom=giso_atom.cpu().numpy(),
            gamma_atom_sum=gamma_atom.sum(-1).cpu().numpy(),
            mode_kappa_P=mode_kP.cpu().numpy(), mode_kappa_C=mode_kC.cpu().numpy(),
            kappa_P_RTA=mode_kP.sum(dim=(1, 2)).cpu().numpy(),
            kappa_C=mode_kC.sum(dim=(1, 2)).cpu().numpy(),
            S_P_atom=self.orbit_mean(SP_raw), S_C_atom=self.orbit_mean(SC_raw),
            S_P_atom_unsym=SP_raw, S_C_atom_unsym=SC_raw,
            gamma_heat_weighted_atom=self.orbit_mean(Gam_raw),
            gamma_heat_weighted=np.einsum("tqm,tqm->t", cv2, g_tot) / cv2.sum(axis=(1, 2)),
            map_atoms=self.map_atoms, num_sampling_grid_points=N,
        )
        res["S_atom"] = res["S_P_atom"] + res["S_C_atom"]
        # --- by-products that need no further triplet pass
        # kappa without isotope scattering: the ph-ph linewidths are the same, only gamma_iso is dropped
        kP0, kC0, _ = self.mode_kappa(gamma, torch.zeros_like(gamma_iso), gv, sum2, N)
        res.update(kappa_P_RTA_noiso=kP0.sum(dim=(1, 2)).cpu().numpy(), kappa_C_noiso=kC0.sum(dim=(1, 2)).cpu().numpy())
        # harmonic mode character at the irreducible points: weight of each mode on each atom, |e_i|^2 (sum_i = 1),
        # and participation ratio 1 / (n_p sum_i w_i^2) (1 = spread over all atoms, 1/n_p = on one atom)
        ev = c.eigvecs[torch.as_tensor(gps, device=dev)]
        wgt = (ev.abs() ** 2).reshape(n_ir, n_p, 3, nb).sum(2).permute(0, 2, 1).to(torch.float64)  # (n_ir, nb, n_p)
        res.update(mode_atom_weight=wgt.cpu().numpy(),
                   participation_ratio=(1.0 / (n_p * (wgt ** 2).sum(-1))).cpu().numpy())
        if self.fcut is not None:
            ach = gatom_ch.cpu().numpy()
            if ach.shape[-1] == 3:  # old parts: decay_LL, decay_other, coalescence_LL computed; the 4th is the rest
                ach = np.concatenate([ach, (res["gamma_atom"] - ach.sum(-1))[..., None]], axis=-1)
                labels = self.ATOM_CHANNELS_OLD
            else:
                labels = self.ATOM_CHANNELS
            res.update(gamma_channel=gamma_ch.cpu().numpy(), gamma_atom_channel=ach, channel_fcut=self.fcut,
                       channel_labels=np.array(self.CHANNELS, dtype="S24"),
                       atom_channel_labels=np.array(labels, dtype="S24"))
        pdos, wpts, wlow = self.local_dos()
        res.update(pdos_atom=pdos, pdos_frequency=wpts, ldos_low_atom=wlow, ldos_low_fmax=4.0)
        res["t_velocity"], res["t_gamma"] = t_vel, t_gamma
        return res

    # ------------------------------------------------------------------
    def star_check(self, is_isotope=True):
        """Verify gamma_{g(i)}(R q) = gamma_i(q) at one star member (validates the orbit mean)."""
        from phono3py.phonon.grid import get_grid_points_by_rotations
        c, s = self.c, self.s
        sym = s.ph3.primitive_symmetry
        rots, perms = sym.dataset.rotations, sym.atomic_permutations
        best = None
        for gp in s.ir_grid_points:
            rmap = get_grid_points_by_rotations(gp, s.bz_grid)
            if len(np.unique(rmap)) > 1:
                best = (gp, rmap)
                break
        if best is None:
            return None
        gp, rmap = best
        ga0, _ = self.gamma_atom_at_gp(gp)
        f0 = c.freqs[gp].cpu().numpy()
        ga0 = c.average_by_degeneracy(ga0.transpose(1, 2), f0).cpu().numpy()  # (nT, n_p, nb)
        ga0 = self.little_group_average(ga0.transpose(0, 2, 1), gp).transpose(0, 2, 1)
        brot = np.array(s.bz_grid.rotations)  # reciprocal rotations acting on grid addresses
        out = []
        for k in range(len(rmap)):
            if rmap[k] == gp:
                continue
            gq = rmap[k]
            R = brot[k]
            # real-space rotation r with (r^T)^-1 = +-R ; time reversal does not permute atoms
            cand = [o for o, r in enumerate(rots)
                    if np.array_equal(np.linalg.inv(r.T).round().astype(int), R)
                    or np.array_equal(-np.linalg.inv(r.T).round().astype(int), R)]
            if not cand:
                continue
            ga1, _ = self.gamma_atom_at_gp(gq)
            ga1 = c.average_by_degeneracy(ga1.transpose(1, 2), c.freqs[gq].cpu().numpy()).cpu().numpy()
            ga1 = self.little_group_average(ga1.transpose(0, 2, 1), gq).transpose(0, 2, 1)
            p = perms[cand[0]]
            scale = np.abs(ga0).max()
            e_fwd = np.abs(ga1[:, p, :] - ga0).max() / scale
            e_inv = np.abs(ga1[:, np.argsort(p), :] - ga0).max() / scale
            e_none = np.abs(ga1 - ga0).max() / scale
            out = dict(gp=int(gp), gp_rot=int(gq), op=int(cand[0]), err_perm=float(e_fwd),
                       err_perm_inverse=float(e_inv), err_unpermuted=float(e_none))
            break
        return out


def find_defect(cell, reference=None, tol=1.0):
    """(index, cartesian position, label) of the defect: minority-species atom, or with a
    reference (perfect) cell of the same lattice the reference site with no atom within tol A."""
    sym = list(cell.symbols)
    uniq, cnt = np.unique(sym, return_counts=True)
    if len(uniq) > 1:
        minority = uniq[np.argmin(cnt)]
        idx = sym.index(minority)
        return idx, cell.positions[idx], f"{minority} on site {idx}"
    if reference is not None:
        for sp in reference.scaled_positions:
            d = cell.scaled_positions - sp
            d -= np.round(d)
            if np.linalg.norm(d @ cell.cell, axis=1).min() > tol:
                return -1, sp @ cell.cell, "vacancy"
    return -1, np.full(3, np.nan), "none"
