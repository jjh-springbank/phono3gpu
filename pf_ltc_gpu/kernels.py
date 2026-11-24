# -*- coding: utf-8 -*-
"""Torch kernels (device-agnostic) for RTA + Wigner thermal conductivity.

Each function is a tensor port of the phono3py 3.12.1 C routine named in its
docstring; constants, signs and masks are copied from that code so that results
agree numerically with phono3py.
"""
import math

import numpy as np
import torch

M_2PI = 6.283185307179586
THZTOEVPARKB = 47.992398658977166
THM_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Phase factors
# ---------------------------------------------------------------------------
def padded_svecs(svecs, multi, device):
    """Pad compact smallest vectors to (n_satom, n_patom, M, 3) + mask.

    multi : (n_satom, n_patom, 2) [count, start].
    """
    n_s, n_p, _ = multi.shape
    M = int(multi[:, :, 0].max())
    idx = multi[:, :, 1][:, :, None] + np.arange(M)[None, None, :]
    mask = np.arange(M)[None, None, :] < multi[:, :, 0][:, :, None]
    idx = np.where(mask, idx, 0)
    v = svecs[idx]  # (n_s, n_p, M, 3)
    return (torch.as_tensor(v, device=device, dtype=torch.float64),
            torch.as_tensor(mask, device=device, dtype=torch.float64),
            torch.as_tensor(multi[:, :, 0], device=device, dtype=torch.float64))


def phase_factors(q, sv, mask, count):
    """get_phase_factor for a batch of q.

    q : (B, 3) reduced.  Returns (B, n_patom, n_satom) complex:
    mean over multiplicity of exp(2 pi i q.svec(s relative to p)).
    """
    ph = M_2PI * torch.einsum("bx,spmx->bspm", q, sv)
    re = (torch.cos(ph) * mask).sum(-1) / count
    im = (torch.sin(ph) * mask).sum(-1) / count
    return torch.complex(re, im).transpose(1, 2).contiguous()  # (B, n_p, n_s)


# ---------------------------------------------------------------------------
# Harmonic phonons
# ---------------------------------------------------------------------------
class DynMat:
    """Dynamical matrix from compact fc2 (phonopy convention, no NAC)."""

    def __init__(self, fc2, svecs, multi, masses, s2pp, device, nac=None):
        self.device = device
        n_p, n_s = fc2.shape[:2]
        self.n_p, self.n_s = n_p, n_s
        self.sv, self.mask, self.count = padded_svecs(svecs, multi, device)
        # gather fc2 into (i, a, s, b) then scatter-sum supercell atoms to primitive j
        self.fc2 = torch.as_tensor(fc2, device=device, dtype=torch.float64)
        self.s2pp = torch.as_tensor(s2pp, device=device, dtype=torch.int64)
        m = torch.as_tensor(masses, device=device, dtype=torch.float64)
        self.inv_sqrt_mm = 1.0 / torch.sqrt(m[:, None] * m[None, :])  # (n_p, n_p)
        # one-hot (n_s, n_p) for s -> primitive j
        self.onehot = torch.zeros((n_s, n_p), device=device, dtype=torch.complex128)
        self.onehot[torch.arange(n_s, device=device), self.s2pp] = 1
        self.nac = None
        if nac is not None:  # Gonze-Lee: fc2 above must be the short-range fc2
            t = lambda a, dt=torch.float64: torch.as_tensor(a, device=device, dtype=dt)
            G = t(nac["G_list"])
            pos = t(nac["positions"])
            dr = pos[:, None, :] - pos[None, :, :]  # (n_p, n_p, 3)
            self.nac = dict(
                G=G, eps=t(nac["dielectric"]), Z=t(nac["born"]), lam=nac["Lambda"],
                factor=nac["factor"], dd_q0=t(nac["dd_q0"], torch.complex128),
                rec=t(nac["rec_lattice"]),
                phase=torch.exp(2j * M_2PI / 2 * torch.einsum("gx,ijx->gij", G, dr).to(torch.complex128)),
            )

    def dipole_dipole(self, q, q_dir=None):
        """Gonze-Lee reciprocal dipole-dipole term (B, nb, nb), mass-scaled.

        Port of phonors get_recip_dipole_dipole (= phonopy C dym_get_recip_dipole_dipole).
        K = q + G with |K|^2 < 1e-10 contributes zero, or d d^T/(d eps d) when the
        approach direction q_dir (reduced, (B, 3)) is given.
        """
        n = self.nac
        qc = q @ n["rec"].T  # cartesian, no 2pi
        K = qc[:, None, :] + n["G"][None]  # (B, nG, 3)
        eps = n["eps"]
        kek = torch.einsum("bga,ac,bgc->bg", K, eps, K)
        zero = (K * K).sum(-1) < 1e-10
        safe = torch.where(zero, torch.ones_like(kek), kek)
        kk = K[..., :, None] * K[..., None, :] / safe[..., None, None] * torch.exp(-safe / (4 * n["lam"] ** 2))[..., None, None]
        if q_dir is not None:
            dc = q_dir @ n["rec"].T
            kd = dc[:, :, None] * dc[:, None, :] / torch.einsum("ba,ac,bc->b", dc, eps, dc)[:, None, None]
            kk = torch.where(zero[..., None, None], kd[:, None], kk)
        else:
            kk = torch.where(zero[..., None, None], torch.zeros_like(kk), kk)
        dd = torch.einsum("bgac,gij->biajc", kk.to(torch.complex128), n["phase"])  # (B, i, a, j, c)
        Zc = n["Z"].to(torch.complex128)
        dd = torch.einsum("imk,jnl,bimjn->bikjl", Zc, Zc, dd)
        ar = torch.arange(self.n_p, device=self.device)
        dd[:, ar, :, ar, :] -= n["dd_q0"][:, None]
        dd = dd * n["factor"] * self.inv_sqrt_mm[None, :, None, :, None]
        nb = 3 * self.n_p
        return dd.reshape(-1, nb, nb)

    def __call__(self, q, q_dir=None):
        """q : (B, 3) reduced coordinates -> (B, 3n_p, 3n_p) complex Hermitian."""
        q = torch.as_tensor(q, device=self.device, dtype=torch.float64).reshape(-1, 3)
        ph = phase_factors(q, self.sv, self.mask, self.count)  # (B, n_p, n_s)
        fcc = self.fc2.to(torch.complex128)
        # D[b, i, a, j, c] = sum_s fc2[i, s, a, c] * ph[b, i, s] * [s2pp[s] == j]
        D = torch.einsum("isac,bis,sj->biajc", fcc, ph, self.onehot)
        D = D * self.inv_sqrt_mm[None, :, None, :, None]
        nb = 3 * self.n_p
        D = D.reshape(-1, nb, nb)
        D = (D + D.conj().transpose(1, 2)) / 2
        if self.nac is not None:
            dd = self.dipole_dipole(q, q_dir)
            D = D + (dd + dd.conj().transpose(1, 2)) / 2
        return D


def solve_phonons(dynmat, qpoints, factor, chunk=None, mem_budget=1e9):
    """Frequencies (THz) and eigenvectors (columns) at qpoints."""
    if chunk is None:
        chunk = max(1, int(mem_budget // (dynmat.n_p * dynmat.n_s * 9 * 16 * 2)))
    freqs, eigvecs = [], []
    for i in range(0, len(qpoints), chunk):
        D = dynmat(qpoints[i:i + chunk])
        w, v = torch.linalg.eigh(D, UPLO="L")
        freqs.append(torch.sqrt(torch.abs(w)) * torch.sign(w) * factor)
        eigvecs.append(v)
    return torch.cat(freqs), torch.cat(eigvecs)


# ---------------------------------------------------------------------------
# Velocity operator (phono3py.phonon.velocity_operator.VelocityOperator)
# ---------------------------------------------------------------------------
def _sqrt_dm(D, flag_gamma):
    w, v = torch.linalg.eigh(D)
    if flag_gamma.any():
        v = v.clone()
        v[flag_gamma, :, 0:3] = 0
    om = torch.einsum("bji,bjk,bkl->bil", v.conj(), D, v)
    om = torch.sqrt(torch.diagonal(om, dim1=1, dim2=2))
    return torch.einsum("bij,bj,bkj->bik", v, om, v.conj())


def velocity_operators(dynmat, q, lattice, factor, q_length=5e-6, dynmat0=None):
    """Velocity operators (B, nb, nb, 3) at reduced q (B, 3).

    With NAC (phono3py-wte VelocityOperator + phonopy DynamicalMatrixNAC.run):
    the eigenvectors at q = Gamma come from the dynamical matrix without NAC and
    with the original fc2 (dynmat0); finite differences around Gamma pass the
    approach direction q +- dq.
    """
    dev = dynmat.device
    q = torch.as_tensor(q, device=dev, dtype=torch.float64).reshape(-1, 3)
    lat = torch.as_tensor(lattice, device=dev, dtype=torch.float64)
    nac = dynmat.nac is not None
    D = dynmat(q)
    if nac and dynmat0 is not None:
        at_g = torch.linalg.norm(q @ dynmat.nac["rec"].T, dim=1) < 1e-5
        if at_g.any():
            D[at_g] = dynmat0(q[at_g])
    _, V = torch.linalg.eigh(D)
    out = []
    for d in range(3):
        dqc = torch.zeros(3, device=dev, dtype=torch.float64)
        dqc[d] = q_length
        dq = lat @ dqc  # reduced
        flag = torch.linalg.norm(q, dim=1) < torch.linalg.norm(dq)
        if nac:
            s2 = _sqrt_dm(dynmat(q + dq, q_dir=q + dq), flag)
            s1 = _sqrt_dm(dynmat(q - dq, q_dir=q - dq), flag)
        else:
            s2 = _sqrt_dm(dynmat(q + dq), flag)
            s1 = _sqrt_dm(dynmat(q - dq), flag)
        ddm = (s2 - s1) / q_length / 2
        g = torch.einsum("bji,bjk,bkl->bil", V.conj(), ddm, V) * factor
        out.append((g.conj().transpose(1, 2) + g) / 2)
    return torch.stack(out, dim=-1)


# ---------------------------------------------------------------------------
# fc3 -> normal coordinates (real_to_reciprocal.c + reciprocal_to_normal.c)
# ---------------------------------------------------------------------------
class Fc3Normal:
    """|Phi(q0 b0, q1 b1, q2 b2)|^2 for batches of triplets.

    Port of r2r_real_to_reciprocal (legacy and r0-average) followed by
    reciprocal_to_normal_squared.  The supercell atoms are reordered so that
    images of each primitive atom are contiguous: s -> (p, c).
    """

    def __init__(self, fc3, svecs, multi, masses, p2s, s2pp, all_shortest,
                 make_r0_average, cutoff_frequency, device, cdtype=torch.complex128,
                 x_budget=2e9, band_block=None):
        self.device = device
        self.x_budget = x_budget
        self.band_block = band_block
        self.cdtype = cdtype
        self.rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
        n_p, n_s = fc3.shape[:2]
        self.n_p, self.n_s = n_p, n_s
        n_c = n_s // n_p
        self.n_c = n_c
        order = np.argsort(s2pp, kind="stable")
        assert (np.bincount(s2pp, minlength=n_p) == n_c).all()
        self.order = order
        self.make_r0_average = make_r0_average
        self.cutoff = cutoff_frequency

        fc3o = fc3[:, order][:, :, order]  # (n_p, n_s, n_s, 3, 3, 3)
        ash = all_shortest[:, order][:, :, order].astype("double")
        if make_r0_average:
            w1 = fc3o * (1 + 2 * ash)[..., None, None, None]
            w23 = fc3o * (1 - ash)[..., None, None, None]
            ws = [w1, w23]
        else:
            ws = [fc3o]
        # (p0, p1, c1, p2, c2, 27)
        self.W = [torch.as_tensor(
            w.reshape(n_p, n_p, n_c, n_p, n_c, 27), device=device, dtype=self.rdtype)
            for w in ws]
        sv, mask, count = padded_svecs(svecs, multi, device)
        self.sv = sv[order]
        self.mask = mask[order]
        self.count = count[order]
        # prephase vector: svecs[multi[p2s[i], 0, 1]]
        self.pre_vec = torch.as_tensor(svecs[multi[p2s, 0, 1]], device=device,
                                       dtype=torch.float64)  # (n_p, 3)
        m = torch.as_tensor(masses, device=device, dtype=torch.float64)
        self.inv_sqrt_m = (1.0 / torch.sqrt(m)).repeat_interleave(3)  # (nb,)

    def _K(self, W, phA, phB):
        """sum_{c1,c2} W[p0,p1,c1,p2,c2,x] phA[b,p0,p1,c1] phB[b,p0,p2,c2].

        Blocked over p0 so the intermediate X stays bounded:
        X[b, p0, p1, c1, p2, x, re/im] costs B*blk*n_s*n_p*27*2 reals.
        """
        B = phA.shape[0]
        n_p, n_s = self.n_p, self.n_s
        isz = 8 if self.rdtype == torch.float64 else 4
        if self.band_block:
            blk = max(1, min(n_p, self.band_block // 3))
        else:
            blk = max(1, min(n_p, int(self.x_budget // max(1, B * n_s * n_p * 27 * 2 * isz * 2))))
        phB_ri = torch.stack((phB.real, phB.imag), dim=-1).to(self.rdtype)  # (B,p0,p2,c2,2)
        phA = phA.to(self.cdtype)
        out = []
        for i0 in range(0, n_p, blk):
            sl = slice(i0, i0 + blk)
            X = torch.einsum("ijcklx,bikld->bijckxd", W[sl], phB_ri[:, sl])
            Xc = torch.complex(X[..., 0], X[..., 1])  # (B,p0,p1,c1,p2,x)
            del X
            out.append(torch.einsum("bijckx,bijc->bijkx", Xc, phA[:, sl]))
        return torch.cat(out, dim=1) if len(out) > 1 else out[0]

    def fc3_reciprocal(self, qv):
        """qv : (B, 3, 3) reduced q of the triplets -> (B, nb, nb, nb) complex."""
        B = qv.shape[0]
        n_p, n_c = self.n_p, self.n_c
        ph = [phase_factors(qv[:, i], self.sv, self.mask, self.count)
              .reshape(B, n_p, n_p, n_c) for i in range(3)]  # (B, p0, p_s, c)
        pre = torch.exp(1j * M_2PI * torch.einsum("px,bx->bp", self.pre_vec,
                                                   qv.sum(dim=1))).to(self.cdtype)
        if self.make_r0_average:
            A1 = self._K(self.W[0], ph[1], ph[2]) * pre[:, :, None, None, None]
            A2 = self._K(self.W[1], ph[0], ph[2]) * pre[:, :, None, None, None]
            A3 = self._K(self.W[1], ph[1], ph[0]) * pre[:, :, None, None, None]
            A1 = A1.reshape(B, n_p, n_p, n_p, 3, 3, 3)
            A2 = A2.reshape(B, n_p, n_p, n_p, 3, 3, 3)
            A3 = A3.reshape(B, n_p, n_p, n_p, 3, 3, 3)
            # F[b, i, l, j, m, k, n]
            F = (A1.permute(0, 1, 4, 2, 5, 3, 6)
                 + A2.permute(0, 2, 5, 1, 4, 3, 6)  # A2[j,i,k,m,l,n]
                 + A3.permute(0, 3, 6, 2, 5, 1, 4)) / 3  # A3[k,j,i,n,m,l]
        else:
            A1 = self._K(self.W[0], ph[1], ph[2]) * pre[:, :, None, None, None]
            F = A1.reshape(B, n_p, n_p, n_p, 3, 3, 3).permute(0, 1, 4, 2, 5, 3, 6)
        nb = 3 * n_p
        return F.reshape(B, nb, nb, nb)

    def _K_prealloc(self, W, phA, phB):
        """Same sum as _K, written block by block into one preallocated array (no list + cat: half the peak)."""
        B = phA.shape[0]
        n_p, n_s = self.n_p, self.n_s
        isz = 8 if self.rdtype == torch.float64 else 4
        if self.band_block:
            blk = max(1, min(n_p, self.band_block // 3))
        else:
            blk = max(1, min(n_p, int(self.x_budget // max(1, B * n_s * n_p * 27 * 2 * isz * 2))))
        phB_ri = torch.stack((phB.real, phB.imag), dim=-1).to(self.rdtype)
        phA = phA.to(self.cdtype)
        out = torch.empty((B, n_p, n_p, n_p, 27), device=self.device, dtype=self.cdtype)
        for i0 in range(0, n_p, blk):
            sl = slice(i0, i0 + blk)
            X = torch.einsum("ijcklx,bikld->bijckxd", W[sl], phB_ri[:, sl])
            Xc = torch.complex(X[..., 0], X[..., 1])
            del X
            out[:, sl] = torch.einsum("bijckx,bijc->bijkx", Xc, phA[:, sl])
            del Xc
        return out

    def contract_partners(self, qv, E):
        """Y[b, cart0, m, n] = sum_{j,k} F[b, cart0, j, k] E[b,1][j, m] E[b,2][k, n] without building F.

        Used by the analysis tools for large cells: fc3_reciprocal holds A1, A2, A3 and F (and temporaries) at once,
        about 7 arrays of nb^3; here each of the three r0-average terms is contracted with the partner eigenvectors
        as soon as it is made, so the peak is about 3-4 arrays. Same result as the F route up to rounding.
        E : (B, 3, nb, nb) mass-scaled eigenvectors [b, t, cart, mode]."""
        B = qv.shape[0]
        n_p, n_c = self.n_p, self.n_c
        nb = 3 * n_p
        ph = [phase_factors(qv[:, i], self.sv, self.mask, self.count)
              .reshape(B, n_p, n_p, n_c) for i in range(3)]
        pre = torch.exp(1j * M_2PI * torch.einsum("px,bx->bp", self.pre_vec,
                                                   qv.sum(dim=1))).to(self.cdtype)
        E1 = E[:, 1].reshape(B, n_p, 3, nb)
        E2 = E[:, 2].reshape(B, n_p, 3, nb)
        if self.make_r0_average:
            # index strings follow fc3_reciprocal: A1[i,j,k,l,m,n], A2[j,i,k,m,l,n], A3[k,j,i,n,m,l] -> F[(i,l),(j,m),(k,n)]
            terms = [(self.W[0], ph[1], ph[2], "bijklmn"), (self.W[1], ph[0], ph[2], "bjikmln"),
                     (self.W[1], ph[1], ph[0], "bkjinml")]
        else:
            terms = [(self.W[0], ph[1], ph[2], "bijklmn")]
        Y = None
        for W, pa, pb, idx in terms:
            A = (self._K_prealloc(W, pa, pb) * pre[:, :, None, None, None]).reshape(B, n_p, n_p, n_p, 3, 3, 3)
            X1 = torch.einsum(f"{idx},bknc->bijlmc", A, E2)  # (B, i, j, l, m, mode2)
            del A
            Yt = torch.einsum("bijlmc,bjmM->bilMc", X1, E1)  # (B, i, l, mode1, mode2)
            del X1
            Y = Yt if Y is None else Y.add_(Yt)
            del Yt
        if len(terms) == 3:
            Y /= 3
        return Y.reshape(B, nb, nb, nb)

    def __call__(self, qv, f, e):
        """|Phi|^2 without unit conversion.

        qv : (B, 3, 3) reduced q; f : (B, 3, nb) THz; e : (B, 3, nb, nb) eigvecs.
        Returns (B, nb, nb, nb) real.
        """
        F = self.fc3_reciprocal(qv)
        E = (e * self.inv_sqrt_m[None, None, :, None]).to(self.cdtype)  # [b, t, cart, mode]
        X = torch.einsum("bijk,bkc->bijc", F, E[:, 2])
        X = torch.einsum("bijc,bjm->bimc", X, E[:, 1])
        X = torch.einsum("bimc,bil->blmc", X, E[:, 0])
        val = (X.real.to(torch.float64) ** 2 + X.imag.to(torch.float64) ** 2)
        ok = f > self.cutoff
        f_safe = torch.where(ok, f, torch.ones_like(f))
        fff = f_safe[:, 0, :, None, None] * f_safe[:, 1, None, :, None] * f_safe[:, 2, None, None, :]
        okm = ok[:, 0, :, None, None] & ok[:, 1, None, :, None] & ok[:, 2, None, None, :]
        return torch.where(okm, val / fff, torch.zeros_like(val))


# ---------------------------------------------------------------------------
# Linear tetrahedron method (tetrahedron_method.c), function 'I'
# ---------------------------------------------------------------------------
def sort_omegas(v):
    """Exact port of the sorting network of sort_omegas().

    v : (..., 4).  Returns sorted (..., 4) and ci (index of original vertex 0).
    """
    v0, v1, v2, v3 = v.unbind(-1)
    i = torch.zeros(v0.shape, dtype=torch.int8, device=v.device)
    c = v0 > v1
    w0 = torch.where(c, v1, v0)
    w1 = torch.where(c, v0, v1)
    i = torch.where(c, torch.ones_like(i), i)
    c = v2 > v3
    w2 = torch.where(c, v3, v2)
    w3 = torch.where(c, v2, v3)
    c = w0 > w2
    n0 = torch.where(c, w2, w0)
    n1 = torch.where(c, w0, w2)
    i = torch.where(c & (i == 0), torch.full_like(i, 4), i)
    c = w1 > w3
    n3 = torch.where(c, w1, w3)
    n2 = torch.where(c, w3, w1)
    i = torch.where(c & (i == 1), torch.full_like(i, 3), i)
    i = torch.where(~c & (i == 1), torch.full_like(i, 5), i)
    c = n1 > n2
    m1 = torch.where(c, n2, n1)
    m2 = torch.where(c, n1, n2)
    i = torch.where(c & (i == 4), torch.full_like(i, 2), i)
    i = torch.where(c & (i == 5), torch.full_like(i, 1), i)
    i = torch.where(~c & (i == 4), torch.full_like(i, 1), i)
    i = torch.where(~c & (i == 5), torch.full_like(i, 2), i)
    return torch.stack((n0, m1, m2, n3), dim=-1), i


def thm_weight_I(omega, vs, ci, vertex_limit=False):
    """sum over tetrahedra of I*g / 6 for sorted vertices.

    vertex_limit=False: open regions as in phono3py 3.12.1 C (omega equal to an
    interior vertex gets zero).  True: half-open regions [v_{r-1}, v_r) as in
    phono3py >= 4.7 (phonors), i.e. the limit from above.

    omega : broadcastable to vs[..., 0];  vs : (..., 24, 4) sorted;  ci : (..., 24).
    Returns (...) summed over the 24 tetrahedra.
    """
    om = omega
    v0, v1, v2, v3 = vs.unbind(-1)

    def f(n, m):
        vn, vm = (v0, v1, v2, v3)[n], (v0, v1, v2, v3)[m]
        d = vn - vm
        small = torch.abs(d) < THM_EPSILON
        return torch.where(small, torch.zeros_like(d),
                           (om - vm) / torch.where(small, torch.ones_like(d), d))

    span = v3 - v0
    span = torch.where(span == 0, torch.ones_like(span), span)
    if vertex_limit:
        r1 = (v0 <= om) & (om < v1)
        r2 = (v1 <= om) & (om < v2)
        r3 = (v2 <= om) & (om < v3)
    else:
        r1 = (v0 < om) & (om < v1)
        r2 = (v1 < om) & (om < v2)
        r3 = (v2 < om) & (om < v3)

    f10, f20, f30 = f(1, 0), f(2, 0), f(3, 0)
    f01, f02, f03 = f(0, 1), f(0, 2), f(0, 3)
    f12, f13, f21, f23, f31, f32 = f(1, 2), f(1, 3), f(2, 1), f(2, 3), f(3, 1), f(3, 2)

    # region 1
    g1 = 3 * f10 * f20 / span
    I1 = torch.where(ci == 0, (f01 + f02 + f03) / 3,
                     torch.where(ci == 1, f10 / 3, torch.where(ci == 2, f20 / 3, f30 / 3)))
    # region 2
    f12_20 = f12 * f20
    f13_21 = f13 * f21
    gg = f12_20 + f21 * f13  # identical for both branches (same products)
    gsafe = torch.where(gg < THM_EPSILON, torch.ones_like(gg), gg)
    g2 = 3 / span * (f12 * f20 + f21 * f13)
    I20 = (f03 + f02 * f12_20 / gsafe) / 3
    I21 = (f12 + f13 * f13_21 / gsafe) / 3
    I22 = (f21 + f20 * f12_20 / gsafe) / 3
    I23 = (f30 + f31 * f13_21 / gsafe) / 3
    I2 = torch.where(ci == 0, I20, torch.where(ci == 1, I21, torch.where(ci == 2, I22, I23)))
    I2 = torch.where(gg < THM_EPSILON, torch.zeros_like(I2), I2)
    # region 3
    g3 = 3 * f13 * f23 / span
    I3 = torch.where(ci == 0, f03 / 3,
                     torch.where(ci == 1, f13 / 3,
                                 torch.where(ci == 2, f23 / 3, (f30 + f31 + f32) / 3)))
    z = torch.zeros_like(g1)
    w = torch.where(r1, I1 * g1, torch.where(r2, I2 * g2, torch.where(r3, I3 * g3, z)))
    return w.sum(-1) / 6


def thm_weights_pruned(omega, vs, ci, max_elems=2 ** 22, vertex_limit=False):
    """Tetrahedron weights for every (omega_j, n) pair, skipping zeros.

    omega : (J,) sampling points; vs : (N, 24, 4) sorted vertices; ci : (N, 24).
    Returns (J, N).  A tetrahedron contributes only when v0 < omega < v3
    (otherwise I*g = 0 exactly), so only those (j, n, tet) are evaluated.
    """
    J, N = omega.shape[0], vs.shape[0]
    out = torch.zeros(J * N, dtype=vs.dtype, device=vs.device)
    fmin = vs[..., 0].amin(dim=-1)
    fmax = vs[..., 3].amax(dim=-1)
    act = (fmin[None, :] <= omega[:, None]) & (omega[:, None] <= fmax[None, :])
    jj, nn = act.nonzero(as_tuple=True)
    step = max(1, max_elems // 24)
    for i in range(0, jj.shape[0], step):
        j, n = jj[i:i + step], nn[i:i + step]
        om = omega[j][:, None]
        v = vs[n]  # (k, 24, 4)
        tet = (v[..., 0] < om) & (om < v[..., 3])
        kk, tt = tet.nonzero(as_tuple=True)
        w = thm_weight_I(om[kk], v[kk, tt][:, None], ci[n][kk, tt][:, None], vertex_limit)
        out.index_add_(0, j[kk] * N + n[kk], w * 6)  # thm_weight_I divides by 6
    return (out / 6).reshape(J, N)


def triplet_integration_weights(f0, f1v, f2v, max_elems=2 ** 22, vertex_limit=False, with_g3=False):
    """Integration weights for ph-ph (tpi_get_integration_weight, tp_type=2).

    f0 : (nb0,) frequency points at q0.
    f1v, f2v : (B, nb, 24, 4) frequencies at tetrahedron vertices around q1
        (+rel) and q2 (-rel).
    Returns g1 = delta(f0-f1-f2), g2 = delta(f0+f1-f2) - delta(f0-f1+f2) each
    (B, nb0, nb, nb); with_g3 also returns g3 = sum of the three deltas
    (tp_type=3 third channel, used by the LBTE collision matrix).
    """
    B, nb = f1v.shape[:2]
    nb0 = f0.shape[0]
    f1 = torch.clamp(f1v, min=0)[:, :, None]  # (B, nb1, 1, 24, 4)
    f2 = torch.clamp(f2v, min=0)[:, None, :]  # (B, 1, nb2, 24, 4)
    fv = torch.stack((f2 - f1, f1 - f2, f1 + f2), dim=1)  # (B, 3, nb1, nb2, 24, 4)
    vs, ci = sort_omegas(fv)
    w = thm_weights_pruned(f0, vs.reshape(-1, 24, 4), ci.reshape(-1, 24), max_elems=max_elems,
                           vertex_limit=vertex_limit)
    w = w.reshape(nb0, B, 3, nb, nb).permute(1, 2, 0, 3, 4)  # (B, 3, nb0, nb1, nb2)
    if with_g3:
        return w[:, 2], w[:, 0] - w[:, 1], w.sum(dim=1)
    return w[:, 2], w[:, 0] - w[:, 1]


def gaussian(x, sigma):
    return 0.3989422804014327 / sigma * torch.exp(-x * x / 2 / sigma / sigma)


def triplet_integration_weights_sigma(f0, f1, f2, sigma, cutoff=-1.0, with_g3=False):
    """tpi_get_integration_weight_with_sigma, tp_type=2.

    f0 : (nb0,), f1, f2 : (B, nb).  Returns (g1, g2) like the tetrahedron version.
    """
    a = f0[None, :, None, None]
    b = f1[:, None, :, None]
    c = f2[:, None, None, :]
    g0 = gaussian(a + b - c, sigma)
    g1 = gaussian(a - b + c, sigma)
    g2 = gaussian(a - b - c, sigma)
    if cutoff > 0:
        zero = ((a + b - c).abs() > cutoff) & ((a - b + c).abs() > cutoff) & \
               ((a - b - c).abs() > cutoff)
        g0 = torch.where(zero, torch.zeros_like(g0), g0)
        g1 = torch.where(zero, torch.zeros_like(g1), g1)
        g2 = torch.where(zero, torch.zeros_like(g2), g2)
    if with_g3:
        return g2, g0 - g1, g0 + g1 + g2
    return g2, g0 - g1


def inv_sinh_occupation(f, temps, cutoff):
    """1/sinh(f/(2 kB T)) for f > cutoff else 0.  f (..., nb) -> (nT, ..., nb)."""
    t = temps.reshape((-1,) + (1,) * f.dim())
    ok = f > cutoff
    fs = torch.where(ok, f, torch.ones_like(f))
    v = 1.0 / torch.sinh(THZTOEVPARKB * fs[None] / (2 * t))
    return torch.where(ok[None], v, torch.zeros_like(v))


def degenerate_ids(freqs, cutoff=1e-4):
    """phonopy get_degenerate_ids: smallest band index of each band's degenerate set.

    freqs (..., nb) ascending.  A new set starts where f[j] - f[j-1] >= cutoff.
    """
    nb = freqs.shape[-1]
    first = torch.ones(freqs.shape, dtype=torch.bool, device=freqs.device)
    first[..., 1:] = (freqs[..., 1:] - freqs[..., :-1]) >= cutoff
    idx = torch.arange(nb, device=freqs.device).expand(freqs.shape)
    return torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), dim=-1).values


def degenerate_average_matrix(ids):
    """Block matrices (..., nb, nb) averaging over each degenerate set given by ids."""
    same = (ids[..., :, None] == ids[..., None, :]).to(torch.float64)
    return same / same.sum(dim=-1, keepdim=True)


def bose_einstein(f, t):
    return 1.0 / (torch.exp(THZTOEVPARKB * f / t) - 1)


def occupations(f, temps, cutoff):
    """set_occupations: n(f, T) for f > cutoff else -1.  f (..., nb) -> (nT, ..., nb)."""
    t = temps.reshape((-1,) + (1,) * f.dim())
    ok = f > cutoff
    fs = torch.where(ok, f, torch.ones_like(f))
    n = bose_einstein(fs[None], t)
    return torch.where(ok[None], n, torch.full_like(n, -1.0))


# ---------------------------------------------------------------------------
# Isotope scattering (iso_get_thm_isotope_scattering_strength)
# ---------------------------------------------------------------------------
def isotope_gamma(f0, e0, freqs_g, eig_g, iw, mass_variances, cutoff, num_grid):
    """gamma_iso at one q-point.

    f0 : (nb0,), e0 : (nb, nb0) eigvecs at q0 (columns), freqs_g : (G, nb),
    eig_g : (G, nb, nb) at regular grid points, iw : (G, nb0, nb).
    """
    n_p = mass_variances.shape[0]
    G, nb = freqs_g.shape
    # overlap[g, j, k, l] = sum_m conj(e0[l m, j]) e[g, l m, k]
    ov = torch.einsum("lmj,glmk->gjkl", e0.reshape(n_p, 3, -1).conj(),
                      eig_g.reshape(G, n_p, 3, nb))
    s = (ov.real ** 2 + ov.imag ** 2) @ mass_variances  # (G, nb0, nb)
    okk = (freqs_g >= cutoff)[:, None, :]
    s = torch.where(okk, s * iw, torch.zeros_like(s))
    gamma = s.sum(dim=(0, 2))
    gamma = gamma * (M_2PI / 4 * f0 * f0 / 2)
    gamma = torch.where(f0 < cutoff, torch.zeros_like(gamma), gamma)
    return gamma / num_grid
