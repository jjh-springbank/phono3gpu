# -*- coding: utf-8 -*-
"""Strain-resolved mode Grueneisen tensors from fc2 + fc3, and their heat-capacity weighted statistics.

    pf-run-gpu analyze gruneisen --config ltc_config.py --outdir out_grun
        [--mesh n1,n2,n3]                 statistics on this regular mesh (default: mesh of the config, all q-points)
        [--qpoints .. | --qpoints-file F | --path ..]   tensors at these q-points instead (no statistics)
        [--fbins f0,f1,..]                frequency-bin edges in THz for the split (default 8 equal bins)
        [--fd-check N]                    finite-difference check at N random q-points (see below)

Definition (same as phono3py.phonon3.gruneisen, which this module ports to the device)
    gamma_{kl}(q j) = - (1 / 2 w^2) <e| dD(q)/d eps_{kl} |e>,   d w / w = - sum_kl gamma_kl eps_kl
    eps_kl : displacement gradient, u_k = eps_kl r_l (q is kept fixed in reduced coordinates)
    dPhi_ab(i, j')/d eps_kl = sum_k' fc3_abc(i, j', k') [delta_ck R_l(k' - i) + X_c,kl(atom of k')]
    X : internal displacement of the atoms under strain (zero in the ion-clamped variant), from fc2:
        X = - Phi(Gamma)^+ sum_N fc2(nu, N) R(N - nu)   (pseudo-inverse over the non-acoustic subspace)
Both variants are written (relaxed = phono3py default, clamped = phono3py ion_clamped=True). With NAC the
frequencies / eigenvectors include it; the strain dependence of the NAC term itself is not included (as phono3py).

Degenerate modes: the diagonal elements of single modes depend on the choice of eigenvectors. Written are
  gruneisen_tensor_*      (nq, nb, 3, 3) average over each degenerate set (invariant), index [k, l] as above
  gruneisen_tensor_*_raw  (nq, nb, 3, 3) diagonal elements as they come (this is what phono3py prints)
and the second moments use the full block: sum_{j' in set} Re(M_x[j, j'] M_y[j', j]) (invariant; equals the
sum of squared frequency shifts of the set for any strain).

Statistics on the mesh (per temperature T, c = mode heat capacity), Voigt index x = xx, yy, zz, yz, xz, xy of the
symmetric strain (tensor components, e.g. gamma_yz = (gamma[y, z] + gamma[z, y]) / 2):
  <variant>/heat_capacity                (nT,)        sum c / N                       eV/K per primitive cell
  <variant>/gruneisen_mean               (nT, 6)      sum c gamma_x / sum c
  <variant>/gruneisen_second_moment      (nT, 6, 6)   sum c gamma_x gamma_y / sum c
  <variant>/gruneisen_variance           (nT, 6, 6)   second moment - mean x mean
  <variant>/cv_by_{band,frequency,atom}             partial sums of c / N  (atom: c |e_i|^2)
  <variant>/cv_gamma_by_{band,frequency,atom}       partial sums of c gamma_x / N        (sum over the split = total)
  <variant>/cv_gamma2_by_{band,frequency,atom}      partial sums of c gamma_x gamma_y / N
  mean = sum_split cv_gamma / heat_capacity,  second moment = sum_split cv_gamma2 / heat_capacity.
  Atom split: gamma(q j) = sum_i gamma_i(q j) with i the atom of the row of dD (real part), like the vertex split
  of atom-scatter; the per-atom second moment is sum Re(M_i M).
  gruneisen_antisymmetric_max : largest |gamma[k, l] - gamma[l, k]| / 2 (zero for a rotationally invariant model).

--fd-check N: at N random q-points the dynamical matrix is rebuilt with fc2 + eps * (fc3 . displacements) where
the displacements of a random strain are made on the host from the supercell atom positions (minimum image,
independent of the device contraction), and gamma is compared with -(f(+eps) - f(-eps)) / (2 eps f).
"""
import json
import time

import numpy as np
import torch

from .. import kernels as K
from ..setup_cpu import log
from . import common as C

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class Gruneisen:
    def __init__(self, ctx):
        self.ctx = ctx
        s, dev = ctx.s, ctx.device
        self.n_p, self.n_s = s.fc3.shape[:2]
        n_p = self.n_p
        self.nb = 3 * n_p
        lat = ctx.lattice
        self.sv, self.mask, self.count = K.padded_svecs(s.svecs, s.multi, dev)  # (n_s, n_p, M, 3)
        R = ((self.sv * self.mask[..., None]).sum(2) / self.count[..., None]) @ lat  # (n_s, n_p, 3) Cartesian
        self.R = R.transpose(0, 1).contiguous()  # [i, k'] vector from primitive atom i to supercell atom k'
        self.X = self._internal_strain()  # (n_p, 3, 3, 3) [atom, c, k, l]
        s2pp = torch.as_tensor(s.s2pp, device=dev)
        m = torch.as_tensor(s.masses, device=dev, dtype=torch.float64)
        self.inv_sqrt_mm = 1.0 / torch.sqrt(m[:, None] * m[None, :])
        self.onehot = torch.zeros((self.n_s, n_p), device=dev, dtype=torch.complex128)
        self.onehot[torch.arange(self.n_s, device=dev), s2pp] = 1
        # dPhi/d eps: (n_p, n_s, 3, 3, 9) for the clamped and the relaxed variant, built atom by atom
        Gc = torch.empty((n_p, self.n_s, 3, 3, 3, 3), device=dev, dtype=torch.float64)
        Gr = torch.empty_like(Gc)
        Xs = self.X[s2pp]  # (n_s, 3, 3, 3)
        for i in range(n_p):
            f3 = torch.as_tensor(s.fc3[i], device=dev, dtype=torch.float64)  # (n_s, n_s, 3, 3, 3)
            Gc[i] = torch.einsum("jkabc,kd->jabcd", f3, self.R[i])
            Gr[i] = Gc[i] + torch.einsum("jkabc,kcgd->jabgd", f3, Xs)
            del f3
        self.G = {"clamped": Gc.reshape(n_p, self.n_s, 3, 3, 9), "relaxed": Gr.reshape(n_p, self.n_s, 3, 3, 9)}
        # 9 -> 6 (symmetric strain)
        V = torch.zeros((6, 9), device=dev, dtype=torch.float64)
        for x, (k, l) in enumerate(VOIGT):
            V[x, 3 * k + l] += 0.5
            V[x, 3 * l + k] += 0.5
        self.V = V

    def _internal_strain(self):
        """X[mu, i, j, k]: displacement i of atom mu per unit eps_jk (phono3py Gruneisen._get_X)."""
        ctx = self.ctx
        s, dev = ctx.s, ctx.device
        dm = ctx.dm
        n_p = dm.n_p
        R2 = ((dm.sv * dm.mask[..., None]).sum(2) / dm.count[..., None]) @ ctx.lattice  # (n_s2, n_p, 3)
        fc2 = torch.as_tensor(s.fc2, device=dev, dtype=torch.float64)  # original fc2 (n_p, n_s2, 3, 3)
        B = torch.einsum("nNlj,Nnk->nljk", fc2, R2)  # (n_p, 3, 3, 3)
        D0 = dm(torch.zeros((1, 3), device=dev, dtype=torch.float64))[0].real
        m = torch.as_tensor(s.dm_masses, device=dev, dtype=torch.float64)
        sm = torch.sqrt(m).repeat_interleave(3)
        # projector on the acoustic subspace (uniform translations in mass-weighted coordinates)
        P = torch.zeros((3 * n_p, 3), device=dev, dtype=torch.float64)
        for a in range(3):
            P[a::3, a] = torch.sqrt(m / m.sum())
        Q = torch.eye(3 * n_p, device=dev, dtype=torch.float64) - P @ P.T
        w, v = torch.linalg.eigh(Q @ ((D0 + D0.T) / 2) @ Q)
        keep = w.abs() > 1e-8 * w.abs().max() if n_p > 1 else torch.zeros_like(w, dtype=torch.bool)
        self.n_internal = int(keep.sum())
        inv = torch.where(keep, 1.0 / torch.where(keep, w, torch.ones_like(w)), torch.zeros_like(w))
        Gam = (v * inv[None]) @ v.T / (sm[:, None] * sm[None, :])  # (3n, 3n)
        X = -(Gam @ B.reshape(3 * n_p, 9)).reshape(n_p, 3, 3, 3)
        return X

    # ------------------------------------------------------------------
    def strain_matrices(self, q, variant="relaxed", atoms=False):
        """f (B, nb), E (B, nb, nb), M (B, nb, nb, 9) [Hermitian in the band pair; M[j, j] = gamma_j], and with
        atoms=True Ma (B, n_p, nb, nb, 9) with sum over atoms = M."""
        ctx = self.ctx
        n_p, nb = self.n_p, self.nb
        q = torch.as_tensor(q, device=ctx.device, dtype=torch.float64).reshape(-1, 3)
        B = len(q)
        f, E = C.phonons_at(ctx, q)
        ph = K.phase_factors(q, self.sv, self.mask, self.count)  # (B, n_p, n_s)
        dD = torch.einsum("isacx,bis,sj->biajcx", self.G[variant].to(torch.complex128), ph, self.onehot)
        dD = (dD * self.inv_sqrt_mm[None, :, None, :, None, None]).reshape(B, nb, nb, 9)
        T = torch.einsum("bIJx,bJs->bIsx", dD, E)  # dD e
        del dD
        w = torch.sign(f) * (f / ctx.s.factor) ** 2
        ok = f.abs() > ctx.s.cutoff_frequency
        den = w[:, :, None] + w[:, None, :]
        okk = ok[:, :, None] & ok[:, None, :] & (den.abs() > 0)
        pref = torch.where(okk, -1.0 / torch.where(okk, den, torch.ones_like(den)), torch.zeros_like(den))
        if atoms:
            Ma = torch.einsum("biaj,biakx->bijkx", E.conj().reshape(B, n_p, 3, nb), T.reshape(B, n_p, 3, nb, 9))
            Ma = Ma * pref[:, None, :, :, None]
            Ma = (Ma + Ma.conj().transpose(2, 3)) / 2
            return f, E, Ma.sum(1), Ma
        M = torch.einsum("bIj,bIkx->bjkx", E.conj(), T) * pref[..., None]
        M = (M + M.conj().transpose(1, 2)) / 2
        return f, E, M, None

    def tensors(self, q, variant="relaxed", chunk=None):
        """(f, raw (B, nb, 3, 3), degenerate-averaged (B, nb, 3, 3)) on the host."""
        chunk = chunk or self._chunk(False)
        fs, raws, avgs = [], [], []
        for i in range(0, len(q), chunk):
            f, _, M, _ = self.strain_matrices(q[i:i + chunk], variant)
            raw = torch.diagonal(M, dim1=1, dim2=2).real.permute(0, 2, 1)  # (B, nb, 9)
            avg = C.degenerate_mean(raw.transpose(1, 2), f[:, None, :]).transpose(1, 2)
            fs.append(f.cpu().numpy()); raws.append(raw.cpu().numpy()); avgs.append(avg.cpu().numpy())
        n = sum(len(x) for x in fs)
        return np.concatenate(fs), np.concatenate(raws).reshape(n, self.nb, 3, 3), np.concatenate(avgs).reshape(n, self.nb, 3, 3)

    def _chunk(self, atoms):
        per = self.nb ** 2 * 9 * 16 * (6 + (3 * self.n_p if atoms else 0)) + self.n_p * self.n_s * 9 * 9 * 16 * 2
        return max(1, int(0.5 * self.ctx.mem_budget // per))

    # ------------------------------------------------------------------
    def statistics(self, q, variant, edges, progress=True):
        """Heat-capacity weighted sums over the q-points q (N, 3) (whole regular mesh). Returns a dict of arrays."""
        ctx = self.ctx
        dev, nb, n_p = ctx.device, self.nb, self.n_p
        nT = len(ctx.temps)
        edges_t = torch.as_tensor(edges, device=dev, dtype=torch.float64)
        nbin = len(edges) - 1
        z = lambda *sh: torch.zeros(sh, device=dev, dtype=torch.float64)
        acc = dict(cv_by_band=z(nT, nb), cv_gamma_by_band=z(nT, nb, 6), cv_gamma2_by_band=z(nT, nb, 6, 6),
                   cv_by_frequency=z(nT, nbin), cv_gamma_by_frequency=z(nT, nbin, 6),
                   cv_gamma2_by_frequency=z(nT, nbin, 6, 6),
                   cv_by_atom=z(nT, n_p), cv_gamma_by_atom=z(nT, n_p, 6), cv_gamma2_by_atom=z(nT, n_p, 6, 6))
        fs, raws, avgs = [], [], []
        asym = 0.0
        N = len(q)
        chunk = self._chunk(True)
        t0 = time.time()
        for i0 in range(0, N, chunk):
            f, E, M9, Ma9 = self.strain_matrices(q[i0:i0 + chunk], variant, atoms=True)
            B = len(f)
            raw = torch.diagonal(M9, dim1=1, dim2=2).real.permute(0, 2, 1)  # (B, nb, 9)
            avg = C.degenerate_mean(raw.transpose(1, 2), f[:, None, :]).transpose(1, 2)
            r33 = avg.reshape(B, nb, 3, 3)
            asym = max(asym, float((r33 - r33.transpose(2, 3)).abs().max()) / 2)
            fs.append(f.cpu().numpy()); raws.append(raw.cpu().numpy()); avgs.append(avg.cpu().numpy())
            M = torch.einsum("xy,bjky->bjkx", self.V.to(M9.dtype), M9)  # (B, nb, nb, 6)
            Ma = torch.einsum("xy,bijky->bijkx", self.V.to(M9.dtype), Ma9)
            del M9, Ma9
            dg = C.degenerate_mask(f).to(M.dtype)  # (B, nb, nb)
            g1 = C.degenerate_mean(torch.diagonal(M, dim1=1, dim2=2).real, f[:, None, :]).transpose(1, 2)  # (B, nb, 6)
            Md = M * dg[..., None]
            g2 = torch.einsum("bjkx,bkjy->bjxy", Md, M).real  # (B, nb, 6, 6)
            ga1 = torch.diagonal(Ma, dim1=2, dim2=3).real.permute(0, 1, 3, 2)  # (B, n_p, nb, 6)
            ga2 = torch.einsum("bijkx,bkjy->bijxy", Ma * dg[:, None, :, :, None], M).real  # (B, n_p, nb, 6, 6)
            del M, Ma, Md
            wgt = (E.abs() ** 2).reshape(B, n_p, 3, nb).sum(2)  # (B, n_p, nb)
            cv = C.mode_heat_capacity(f, ctx.temps, ctx.s.cutoff_frequency)  # (nT, B, nb)
            oh = ((f[:, :, None] >= edges_t[None, None, :-1]) & (f[:, :, None] < edges_t[None, None, 1:])).to(torch.float64)
            acc["cv_by_band"] += cv.sum(1)
            acc["cv_gamma_by_band"] += torch.einsum("tbj,bjx->tjx", cv, g1)
            acc["cv_gamma2_by_band"] += torch.einsum("tbj,bjxy->tjxy", cv, g2)
            acc["cv_by_frequency"] += torch.einsum("tbj,bjn->tn", cv, oh)
            acc["cv_gamma_by_frequency"] += torch.einsum("tbj,bjn,bjx->tnx", cv, oh, g1)
            acc["cv_gamma2_by_frequency"] += torch.einsum("tbj,bjn,bjxy->tnxy", cv, oh, g2)
            acc["cv_by_atom"] += torch.einsum("tbj,bij->ti", cv, wgt)
            acc["cv_gamma_by_atom"] += torch.einsum("tbj,bijx->tix", cv, ga1)
            acc["cv_gamma2_by_atom"] += torch.einsum("tbj,bijxy->tixy", cv, ga2)
            if progress and (i0 // chunk) % max(1, (N // chunk) // 5) == 0:
                log(f"[GRUN] {variant}: {min(i0 + chunk, N)}/{N} q-points ({time.time() - t0:.1f} s)")
        out = {k: (v / N).cpu().numpy() for k, v in acc.items()}
        ctot = out["cv_by_band"].sum(1)
        mean = out["cv_gamma_by_band"].sum(1) / ctot[:, None]
        m2 = out["cv_gamma2_by_band"].sum(1) / ctot[:, None, None]
        out.update(heat_capacity=ctot, gruneisen_mean=mean, gruneisen_second_moment=m2,
                   gruneisen_variance=m2 - mean[:, :, None] * mean[:, None, :],
                   gruneisen_antisymmetric_max=asym)
        n = sum(len(x) for x in fs)
        return out, np.concatenate(fs), np.concatenate(raws).reshape(n, nb, 3, 3), np.concatenate(avgs).reshape(n, nb, 3, 3)

    # ------------------------------------------------------------------
    def host_displacement_vectors(self):
        """R_host[i, k'] (n_p, n_s, 3): Cartesian vector from primitive atom i to supercell atom k', mean over the
        equidistant minimum images, from the supercell positions (independent of svecs / multi)."""
        s = self.ctx.s
        sc = s.ph3.supercell
        sp, L = np.array(sc.scaled_positions), np.array(sc.cell)
        sh = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype="double")
        out = np.zeros((self.n_p, self.n_s, 3))
        for i, p in enumerate(s.p2s):
            d = sp - sp[p]
            d -= np.round(d)
            cand = (d[:, None, :] + sh[None]) @ L  # (n_s, 27, 3)
            n = np.linalg.norm(cand, axis=-1)
            m = n < n.min(axis=1, keepdims=True) + 1e-5
            out[i] = (cand * m[..., None]).sum(1) / m.sum(1)[:, None]
        return out

    def fd_check(self, q, variant="relaxed", eps=1e-5, seed=0):
        """Finite-difference check at q (B, 3) for one random strain. Returns a dict of numbers."""
        ctx = self.ctx
        s, dev = ctx.s, ctx.device
        rng = np.random.default_rng(seed)
        Ev = rng.normal(size=(3, 3))
        Ev = (Ev + Ev.T) / 2
        Rh = self.host_displacement_vectors()
        X = self.X.cpu().numpy() if variant == "relaxed" else np.zeros((self.n_p, 3, 3, 3))
        dphi = np.zeros((self.n_p, self.n_s, 3, 3))
        for i in range(self.n_p):
            u = Rh[i] @ Ev.T + np.einsum("kcgd,gd->kc", X[s.s2pp], Ev)  # (n_s, 3): u_c = E_cl R_l + X_c,kl E_kl
            dphi[i] = np.einsum("jkabc,kc->jab", s.fc3[i], u)
        dmd = K.DynMat(dphi, s.svecs, s.multi, s.masses, s.s2pp, dev)
        q = torch.as_tensor(q, device=dev, dtype=torch.float64).reshape(-1, 3)
        D = C.dynmat_at(ctx.dm, q)
        dD = dmd(q)
        fp = torch.linalg.eigvalsh(D + eps * dD)
        fm = torch.linalg.eigvalsh(D - eps * dD)
        sq = lambda w: torch.sqrt(torch.abs(w)) * torch.sign(w) * s.factor
        fp, fm = sq(fp), sq(fm)
        f, _, M, _ = self.strain_matrices(q, variant)
        Et = torch.as_tensor(Ev.reshape(9), device=dev, dtype=torch.complex128)
        Me = torch.einsum("bjkx,x->bjk", M, Et)
        f_h, fp, fm, Me = f.cpu().numpy(), fp.cpu().numpy(), fm.cpu().numpy(), Me.cpu().numpy()
        from phonopy.phonon.degeneracy import degenerate_sets
        err, err_diag, gmax, nmode = 0.0, 0.0, 0.0, 0
        for b in range(len(q)):
            for dset in degenerate_sets(f_h[b]):
                if abs(f_h[b, dset[0]]) < max(s.cutoff_frequency, 1e-3):
                    continue
                f0 = f_h[b, dset]
                lam = np.linalg.eigvalsh(Me[b][np.ix_(dset, dset)])  # ascending
                fd = 0.5 * (np.sort(-(fp[b, dset] - f0) / (eps * f0)) + np.sort((fm[b, dset] - f0) / (eps * f0)))
                err = max(err, np.abs(lam - fd).max())
                err_diag = max(err_diag, abs(np.real(np.trace(Me[b][np.ix_(dset, dset)])) / len(dset) - fd.mean()))
                gmax = max(gmax, np.abs(fd).max())
                nmode += len(dset)
        return dict(variant=variant, eps=eps, n_q=int(len(q)), n_modes=nmode, max_abs_gamma_fd=float(gmax),
                    max_abs_diff=float(err), max_abs_diff_set_mean=float(err_diag))


# ---------------------------------------------------------------------------
def add_parser(sub):
    a = sub.add_parser("gruneisen", help="strain-resolved mode Grueneisen tensors and their mean / variance split",
                       description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    C.add_common_args(a, "out_gruneisen")
    C.add_q_args(a)
    a.add_argument("--mesh", default=None, metavar="n1,n2,n3", help="mesh for the statistics (default: config mesh)")
    a.add_argument("--fbins", default=None, metavar="f0,f1,..", help="frequency-bin edges (THz) of the split")
    a.add_argument("--fd-check", type=int, default=0, metavar="N", help="finite-difference check at N random q-points")
    a.set_defaults(func=main)


def main(args):
    total0 = time.time()
    C.resolve_config(args)
    mesh = [int(x) for x in args.mesh.split(",")] if args.mesh else "config"
    ctx = C.make_context(args, "gruneisen", mesh=mesh, need_conductivity=False)
    s = ctx.s
    t0 = time.time()
    g = Gruneisen(ctx)
    ctx.sync()
    t_prep = time.time() - t0
    log(f"[GRUN] dPhi/d eps ready ({t_prep:.2f} s); internal-strain modes: {g.n_internal}; "
        f"max |X| = {float(g.X.abs().max()):.4f} A per unit strain")
    q_user, dist, _ = C.parse_qpoints(args, s.lattice)
    data, units, chk = dict(C.structure_datasets(s)), {}, {}
    data["internal_strain_displacement"] = g.X.cpu().numpy()
    units["internal_strain_displacement"] = "Angstrom per unit strain, [atom, c, k, l]"
    t0 = time.time()
    if q_user is not None:
        for v in ("relaxed", "clamped"):
            f, raw, avg = g.tensors(q_user, v)
            data[f"gruneisen_tensor_{v}"], data[f"gruneisen_tensor_{v}_raw"] = avg, raw
            data[f"gruneisen_{v}"] = np.einsum("qjkk->qj", avg) / 3
        data.update(qpoint=q_user, frequency=f, distance=dist)
        name = "gruneisen-qpoints.hdf5"
    else:
        qg = s.qpoints_bz[np.array(s.bz_grid.grg2bzg, dtype="int64")]
        fmax = None
        for v in ("relaxed", "clamped"):
            if args.fbins:
                edges = np.array([float(x) for x in args.fbins.split(",")])
            else:
                if fmax is None:
                    fmax = float(C.phonons_at(ctx, qg, eigvecs=False).max())
                edges = np.linspace(0.0, fmax * (1 + 1e-9), 9)
            st, f, raw, avg = g.statistics(qg, v, edges)
            for k, val in st.items():
                data[f"{v}/{k}"] = val
            data[f"gruneisen_tensor_{v}"], data[f"gruneisen_tensor_{v}_raw"] = avg, raw
            data[f"gruneisen_{v}"] = np.einsum("qjkk->qj", avg) / 3
            chk[f"{v}_antisymmetric_max"] = float(st["gruneisen_antisymmetric_max"])
            chk[f"{v}_split_sum_band_vs_atom_rel"] = float(
                np.abs(st["cv_gamma2_by_band"].sum(1) - st["cv_gamma2_by_atom"].sum(1)).max()
                / np.abs(st["cv_gamma2_by_band"].sum(1)).max())
            chk[f"{v}_split_sum_band_vs_frequency_rel"] = float(
                np.abs(st["cv_gamma2_by_band"].sum(1) - st["cv_gamma2_by_frequency"].sum(1)).max()
                / np.abs(st["cv_gamma2_by_band"].sum(1)).max())
            for it, T in enumerate(s.temperatures):
                m, var = st["gruneisen_mean"][it], st["gruneisen_variance"][it]
                log(f"[GRUN] {v} T={T:.0f} K: <gamma_xx,yy,zz> = {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}; "
                    f"<gamma_yz,xz,xy> = {m[3]:.2e} {m[4]:.2e} {m[5]:.2e}; var diag = "
                    + " ".join(f"{var[x, x]:.4f}" for x in range(6)))
        data.update(qpoint=qg, frequency=f, temperature=s.temperatures, frequency_bin_edges=edges,
                    mesh=np.array(s.bz_grid.D_diag, dtype="int64"))
        mm = data["mesh"]
        name = f"gruneisen-m{mm[0]}x{mm[1]}x{mm[2]}.hdf5"
    ctx.sync()
    t_run = time.time() - t0
    if args.fd_check:
        rng = np.random.default_rng(1)
        qr = rng.uniform(-0.5, 0.5, size=(args.fd_check, 3))
        for v in ("relaxed", "clamped"):
            for eps in (1e-4, 1e-5, 1e-6):
                chk[f"fd_check_{v}_eps{eps:g}"] = g.fd_check(qr, v, eps=eps)
    chk.update(t_prepare_s=t_prep, t_run_s=t_run, gpu_peak_GB=ctx.peak_gb())
    for k, v in chk.items():
        log(f"[CHECK] {k} = {v}")
    for k in list(data):
        if k.startswith("gruneisen_tensor"):
            units[k] = "dimensionless, [q, band, k, l] for displacement gradient eps_kl"
    units.update(frequency="THz", qpoint="reduced")
    for v in ("relaxed", "clamped"):
        units[f"{v}/heat_capacity"] = "eV/K per primitive cell"
    C.write_hdf5(f"{args.outdir}/{name}", data, attrs=dict(checks_json=json.dumps(chk), config=str(args.config)),
                 units=units)
    log(f"=== [FINISH] analyze gruneisen total {time.time() - total0:.2f} s")
    return 0
