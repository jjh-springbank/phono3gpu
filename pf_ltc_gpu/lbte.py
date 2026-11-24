# -*- coding: utf-8 -*-
"""Direct solution of the linearised BTE (phono3py LBTE, irreducible collision matrix).

Port of phono3py 4.10 conductivity/collision_matrix_kernel.py
(IrreducibleCollisionMatrixKernel) and phonon3/collision_matrix.py, kappa-star on,
reducible_collision_matrix off, pinv_solver eigendecomposition.

Layout of the collision matrix (as in phono3py):
    Omega[T, i0, j0, a, i, j, b]   i0, i: irreducible grid points, j: band, a, b: Cartesian.
Each row block i0 is filled while the linewidth kernel runs at grid point i0
(GPUConductivity.gamma_at_gp(..., collision=True)); everything after that is dense
tensor algebra on the device: diagonal, k-star weights, degeneracy averaging,
symmetrisation, eigendecomposition, and the kappa contraction.
"""
import numpy as np
import torch
from phono3py.phonon.grid import get_grid_points_by_rotations
from phono3py.phonon3.triplets import get_triplets_at_q
from phonopy import units as U

from . import kernels as K


def unit_to_WmK():
    """phono3py conductivity.utils.get_unit_to_WmK (2pi from the lifetime definition)."""
    return (U.THz * U.Angstrom) ** 2 / U.Angstrom ** 3 * U.EV / U.THz / (2 * np.pi)


def gp2tp_map(map_triplets, map_q):
    """Vectorised CollisionMatrix._get_gp2tp_map: (gp2tp, swapped) over the GR grid.

    gp2tp[gp1]: index of the irreducible triplet that represents q1 = gp1.
    swapped[gp1]: q1 and q2 of that triplet are exchanged for gp1.
    """
    n = len(map_triplets)
    gp = np.arange(n)
    irq = map_q == gp
    rep = irq & (map_triplets == gp)
    gp2tp = -np.ones(n, dtype="int64")
    gp2tp[rep] = np.arange(rep.sum())
    swapped = np.zeros(n, dtype=bool)
    sw = irq & ~rep
    gp2tp[sw] = gp2tp[map_triplets[sw]]
    swapped[sw] = True
    other = ~irq
    gp2tp[other] = gp2tp[map_q[other]]
    swapped[other] = swapped[map_q[other]]
    assert (gp2tp >= 0).all(), "triplet map inconsistent"
    return gp2tp, swapped


class CollisionMatrix:
    """Irreducible collision matrix of all temperatures, filled row block by row block."""

    def __init__(self, cond, store=None):
        """store: where the (nT, N, N) matrix lives while rows are filled.

        Default: the compute device if all temperatures fit in half of its free
        memory, otherwise host memory (pinned on CUDA); solve() then moves one
        temperature at a time to the device.
        """
        s = cond.s
        self.c = cond
        self.s = s
        self.dev = cond.device
        bz = s.bz_grid
        self.ir = np.asarray(s.ir_grid_points)
        self.rot_bz = np.array([get_grid_points_by_rotations(gp, bz) for gp in self.ir], dtype="int64")
        self.rot_gr = np.asarray(bz.bzg2grg)[self.rot_bz]
        self.R = torch.as_tensor(np.asarray(s.rotations_cartesian), device=self.dev, dtype=torch.float64)
        n_rot = self.rot_bz.shape[1]
        self.weights = np.array([np.sqrt(len(np.unique(r)) / n_rot) for r in self.rot_bz])
        n_ir, nb, nT = len(self.ir), cond.nb, len(cond.temps)
        self.n_ir, self.nb, self.nT = n_ir, nb, nT
        nbytes = nT * (n_ir * nb * 3) ** 2 * 8
        # device peak of solve() for one temperature: eigh input + eigenvector output + cuSOLVER
        # syevd workspace (~4 N^2 doubles, measured: 52.0 GiB at N = 41760) = ~6 N^2 doubles.
        solve_bytes = 6 * (n_ir * nb * 3) ** 2 * 8
        if store is None:
            store = self.dev
            if self.dev.type == "cuda" and nbytes + solve_bytes > 0.9 * torch.cuda.mem_get_info(self.dev)[0]:
                store = torch.device("cpu")
        self.store = torch.device(store)
        print(f"[LBTE] collision matrix N={n_ir * nb * 3}, all T {nbytes / 2 ** 30:.1f} GiB, "
              f"solve peak ~{solve_bytes / 2 ** 30:.1f} GiB -> stored on {self.store}", flush=True)
        self.omega = torch.zeros((nT, n_ir, nb, 3, n_ir, nb, 3), device=self.store, dtype=torch.float64,
                                 pin_memory=(self.store.type == "cpu" and self.dev.type == "cuda"))
        self._rot_gr_t = torch.as_tensor(self.rot_gr, device=cond.device)

    def add_row(self, i0, gp, A, Bm):
        """Row block of irreducible point i0 (BZ index gp) from the per-triplet sums."""
        _, _, map_t, map_q = get_triplets_at_q(gp, self.s.bz_grid)
        g2t, sw = gp2tp_map(np.asarray(map_t), np.asarray(map_q))
        dev = A.device
        idx = torch.as_tensor(g2t, device=dev)
        swt = torch.as_tensor(sw, device=dev)[:, None, None, None]
        M = torch.where(swt, Bm[idx], A[idx])  # (N_q, nT, nb0, nb)
        Mr = M[self._rot_gr_t.to(dev)]  # (n_ir, n_rot, nT, nb0, nb)
        row = torch.einsum("irtjk,rab->tjaikb", Mr, self.R.to(dev))
        # blocking copy: with non_blocking=True the D2H copy into a temporary was still in flight when it was
        # written into the (strided) host slice, which left wrong rows when the matrix is stored on the host.
        self.omega[:, i0].copy_(row)

    # ------------------------------------------------------------------
    def solve(self, gamma, gamma_iso, gv, pinv_cutoff=1e-8, pinv_method=0, boundary=None):
        """Finish the matrix and return kappa (LBTE) and kappa_RTA, each (nT, 6).

        gamma (nT, n_ir, nb), gamma_iso (n_ir, nb) or None, gv (n_ir, nb, 3).
        """
        s, c = self.s, self.c
        dev = self.dev
        n_ir, nb, nT = self.n_ir, self.nb, self.nT
        N = n_ir * nb * 3
        R = self.R
        diag = gamma.to(dev).clone()
        if gamma_iso is not None:
            diag = diag + gamma_iso.to(dev)[None]
        if boundary is not None:
            diag = diag + boundary.to(dev)[None]
        # main diagonal: sum of the rotations that leave the ir point invariant
        self_rot = torch.as_tensor(self.rot_bz == self.ir[:, None], device=dev, dtype=torch.float64)
        Rself = torch.einsum("ir,rab->iab", self_rot, R)  # (n_ir, 3, 3)
        m = torch.arange(n_ir * nb, device=dev)
        w = torch.as_tensor(self.weights, device=dev, dtype=torch.float64)
        ww = w[:, None, None, None, None, None] * w[None, None, None, :, None, None]
        Dg = c.degenerate_average_matrices()[torch.as_tensor(self.ir, device=c.device)].to(dev)

        def finish(it):
            """Omega at temperature it, on the device: diagonal, weights, averaging, symmetrised."""
            om = self.omega[it].to(dev)  # (n_ir, nb, 3, n_ir, nb, 3)
            om5 = om.view(n_ir * nb, 3, n_ir * nb, 3)
            Dm = (diag[it][..., None, None] * Rself[:, None]).reshape(n_ir * nb, 3, 3)
            om5[m, :, m, :] = om5[m, :, m, :] + Dm
            om = om * ww
            # average rows and columns over degenerate bands (phono3py does this always)
            om = torch.einsum("ijJ,iJaxkb->ijaxkb", Dg, om)
            om = torch.einsum("xkK,ijaxKb->ijaxkb", Dg, om)
            om = om.reshape(N, N)
            return (om + om.T) / 2

        freqs = c.freqs[torch.as_tensor(self.ir, device=c.device)].to(dev)  # (n_ir, nb)
        gv = gv.to(dev)
        cutoff = s.cutoff_frequency
        conv = unit_to_WmK() / s.volume
        nq = float(np.prod(s.bz_grid.D_diag))
        addresses = np.asarray(s.bz_grid.addresses)[:n_ir]
        skip = torch.ones((n_ir, nb), dtype=torch.bool, device=dev)
        zero_adr = torch.as_tensor((addresses == 0).all(axis=1), device=dev)
        skip[:, :3] &= ~zero_adr[:, None]  # phono3py skips 3 lowest modes where address==0
        kappa = torch.zeros((nT, 6), dtype=torch.float64, device=dev)
        kappa_rta = torch.zeros_like(kappa)
        eigvals = []
        idx6 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        for it, T in enumerate(c.temps.tolist()):
            if T <= 0:
                continue
            x = freqs * U.THzToEv / (2 * U.Kb * T)
            ok = freqs > cutoff
            inv_sinh = torch.where(ok, 1 / torch.sinh(torch.where(ok, x, torch.ones_like(x))),
                                   torch.zeros_like(x))
            ff = freqs * U.THzToEv * inv_sinh / (4 * U.Kb * T ** 2)
            X = gv * w[:, None, None] * ff[..., None]  # (n_ir, nb, 3)
            # RTA from the main diagonal
            Yr = torch.where(ok[..., None], X / torch.where(ok, diag[it], torch.ones_like(diag[it]))[..., None],
                             torch.zeros_like(X))
            # LBTE: Y = Omega^+ X by eigendecomposition
            A = finish(it)
            if dev.type == "cuda":
                torch.cuda.empty_cache()  # release finish() temporaries before the ~4 N^2 syevd workspace
            ev, V = torch.linalg.eigh(A)
            del A
            eigvals.append(ev)
            keep = (ev.abs() if pinv_method == 0 else ev) > pinv_cutoff
            e = torch.where(keep, 1 / torch.where(keep, ev, torch.ones_like(ev)), torch.zeros_like(ev))
            Y = (V @ (e * (V.T @ X.reshape(-1)))).reshape(n_ir, nb, 3)
            for out, YY in ((kappa, Y), (kappa_rta, Yr)):
                S = torch.einsum("rab,ijb,rcd,ijd->ijac", R, X, R, YY)
                S = S + S.transpose(-1, -2)
                S = torch.where(skip[..., None, None], S, torch.zeros_like(S))
                mk = torch.stack([S[..., a, b] for a, b in idx6], dim=-1) * conv * U.Kb * T ** 2
                out[it] = mk.sum(dim=(0, 1)) / nq
            del V
        self.omega = None
        self.eigenvalues = eigvals
        return kappa, kappa_rta
