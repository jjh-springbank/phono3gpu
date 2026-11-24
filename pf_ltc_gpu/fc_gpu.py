# -*- coding: utf-8 -*-
"""fc2 / fc3 from displacement forces (mmap) on the GPU.

Port of what pf-run does with ``produce_fc2(symmetrize_fc2=True, is_compact_fc=True)``
and ``produce_fc3(is_compact_fc=True)`` (phonopy FDFCSolver + phono3py traditional
fc3: _get_fc3_least_atoms -> distribute_fc3 -> _cutoff_fc3_for_cutoff_pairs).

Host work is integer symmetry bookkeeping (op tables, which atom maps to which);
all force-constant arithmetic (pinv solves, rotations, distribution, cutoff
symmetrisation) runs on ``device``.  The full (N, N, N) fc3 of the cut-off path
is never formed: every element needed for the compact output is generated from
the least-atom rows through a composed (row, atom map, rotation) per first index.

Symmetry operations are those of ``phonopy.Symmetry``.  An operation k with
perm[k, a] == a is a site symmetry of atom a and perm[k] is its rotation about a
(``positions[perm[k][i]] == R x_i + t``), which is what phonopy computes with
shifted positions and zero translation.
"""
import time

import numpy as np
import torch

from .setup_cpu import log

# bytes allowed for one block of gathered forces / rotated delta-fc2 (tunable)
BLOCK_BYTES = 1.5e9


class _Ops:
    """Space group of a supercell as tensors."""

    def __init__(self, cell, symmetry, device):
        rots = np.array(symmetry.symmetry_operations["rotations"], dtype="intc")
        self.rots = rots
        self.perms = np.array(symmetry.atomic_permutations, dtype="int64")
        self.invperms = np.argsort(self.perms, axis=1)
        L = np.array(cell.cell.T, dtype="double")  # column vectors
        Linv = np.linalg.inv(L)
        self.rc = np.array([L @ r @ Linv for r in rots], dtype="double")  # Cartesian R
        self.pure_trans = np.array([(r == np.eye(3, dtype=int)).all() for r in rots])
        self.symprec = symmetry.tolerance
        self.cell = cell
        self.device = device
        self.t_rc = torch.as_tensor(self.rc, device=device)
        self.t_perms = torch.as_tensor(self.perms, device=device)
        self.t_invperms = torch.as_tensor(self.invperms, device=device)

    def site_ops(self, atom):
        """Indices of operations fixing ``atom``, in operation order (= get_site_symmetry)."""
        return np.nonzero(self.perms[:, atom] == atom)[0]

    def reduce_by_direction(self, ops, disp):
        """get_reduced_site_symmetry: keep R with R d = d (fractional d)."""
        direction = np.dot(disp, np.linalg.inv(self.cell.cell))
        keep = [k for k in ops
                if (abs(direction - np.dot(direction, self.rots[k].T)) < self.symprec).all()]
        return np.array(keep, dtype="int64")


def _sym_mappings(perms_sub, done):
    """_get_sym_mappings_from_permutations: first op (in given order) sending atom into done."""
    isdone = np.zeros(perms_sub.shape[1], dtype=bool)
    isdone[np.asarray(done)] = True
    hit = isdone[perms_sub]  # (n_ops, N)
    first = np.argmax(hit, axis=0)
    if not hit.any(axis=0).all():
        raise ValueError("Input forces are not enough to calculate force constants.")
    atoms = perms_sub[first, np.arange(perms_sub.shape[1])]
    return atoms, first


def solve_rows(ops, uvec, fsets, op_idx, f_idx, valid):
    """Batched _solve_force_constants_svd.

    Row b: U_b = [R_k u for (u, k) in its rows], pinv over M rows, and
    fc[b, i] = -pinv(U_b) @ [F_f[invperm_k[i]] R_k^T rows].
    uvec (B, M, 3) displacement per row, op_idx/f_idx (B, M) op and force-set
    index per row, valid (B, M) mask.  fsets (K, N, 3).  Returns (B, N, 3, 3).
    """
    rc = ops.t_rc[op_idx]  # (B, M, 3, 3)
    U = torch.einsum("bmij,bmj->bmi", rc, uvec) * valid[..., None]
    P = torch.linalg.pinv(U)  # (B, 3, M); zero rows -> zero columns
    inv = ops.t_invperms[op_idx]  # (B, M, N)
    F = fsets[f_idx[..., None], inv]  # (B, M, N, 3)
    comb = torch.einsum("bmng,bmhg->bmnh", F, rc) * valid[..., None, None]
    return -torch.einsum("bam,bmnh->bnah", P, comb)


def distribute_rows(fc, targets, map_atoms, map_ops, ops, row_of):
    """phonoc.distribute_fc2: fc[t, k] = R^T fc[d, perm[k]] R (in place, targets only).

    row_of maps atom index -> row of ``fc``.  Atoms that map to themselves are skipped.
    """
    t = np.asarray(targets)
    d = map_atoms[t]
    keep = d != t
    t, d, k = t[keep], d[keep], map_ops[t[keep]]
    if len(t) == 0:
        return fc
    rows_d = torch.as_tensor(row_of[d], device=fc.device)
    perm = ops.t_perms[torch.as_tensor(k, device=fc.device)]  # (T, N)
    src = fc[rows_d[:, None], perm]  # (T, N, 3, 3)
    R = ops.t_rc[torch.as_tensor(k, device=fc.device)]
    fc[torch.as_tensor(row_of[t], device=fc.device)] = torch.einsum(
        "tma,tkmn,tnb->tkab", R, src, R)
    return fc


def fc_rows_from_first_atoms(ops, first_atoms, disps, forces, N, device, rows_for):
    """FDFCSolver._get_force_constants_disps for every displaced atom.

    first_atoms (n,) atom per displacement, disps (n, 3), forces (n, N, 3) tensor.
    rows_for: dict atom -> output row.  Returns dict atom -> (N, 3, 3).
    """
    out = {}
    for a in np.unique(first_atoms):
        sel = np.nonzero(first_atoms == a)[0]
        sops = ops.site_ops(a)
        nd, ns = len(sel), len(sops)
        op_idx = torch.as_tensor(np.tile(sops, nd), device=device)[None]
        f_idx = torch.as_tensor(np.repeat(np.arange(nd), ns), device=device)[None]
        u = torch.as_tensor(np.repeat(disps[sel], ns, axis=0), device=device)[None]
        valid = torch.ones(op_idx.shape, dtype=torch.float64, device=device)
        out[a] = solve_rows(ops, u, forces[torch.as_tensor(sel, device=device)],
                            op_idx, f_idx, valid)[0]
    return out


class FCDataset:
    """Displacement dataset of phono3py_disp.yaml flattened to arrays (host)."""

    def __init__(self, dataset):
        first = dataset["first_atoms"]
        self.n_first = len(first)
        self.first_atom = np.array([x["number"] for x in first], dtype="int64")
        self.first_disp = np.array([x["displacement"] for x in first], dtype="double")
        sec_first, sec_atom, sec_disp, sec_incl = [], [], [], []
        self.sec_range = np.zeros((self.n_first, 2), dtype="int64")
        count = 0
        for i, x in enumerate(first):
            n = len(x["second_atoms"])
            self.sec_range[i] = (count, count + n)
            for y in x["second_atoms"]:
                sec_atom.append(y["number"])
                sec_disp.append(y["displacement"])
                sec_incl.append(y.get("included", None))
            sec_first += [i] * n
            count += n
        self.sec_atom = np.array(sec_atom, dtype="int64")
        self.sec_disp = np.array(sec_disp, dtype="double").reshape(-1, 3)
        self.sec_incl = sec_incl
        self.n_snap = self.n_first + count
        self.cutoff = dataset.get("cutoff_distance")
        self.second_atoms = [x["second_atoms"] for x in first]

    def force_index_second(self, i):
        """Rows of the fc3 force array holding the second displacements of first disp i."""
        a, b = self.sec_range[i]
        return self.n_first + a, self.n_first + b


def _load_forces(mm, rows, device):
    a, b = rows
    return torch.as_tensor(np.asarray(mm[a:b], dtype="double"), device=device)


# ---------------------------------------------------------------------------
# fc2
# ---------------------------------------------------------------------------
def produce_fc2_compact(ph3, fc2_mmap, device, symmetrize=True):
    """produce_fc2(symmetrize_fc2, is_compact_fc=True) from the fc2 force mmap."""
    cell = ph3.phonon_supercell
    prim = ph3.phonon_primitive
    ops = _Ops(cell, ph3.phonon_supercell_symmetry, device)
    ds = ph3.phonon_dataset if ph3.phonon_dataset is not None else ph3.dataset
    first = ds["first_atoms"]
    atoms = np.array([x["number"] for x in first], dtype="int64")
    disps = np.array([x["displacement"] for x in first], dtype="double")
    N = len(cell)
    forces = _load_forces(fc2_mmap, (0, len(first)), device)
    p2s = np.array(prim.p2s_map, dtype="int64")
    row_of = -np.ones(N, dtype="int64")
    row_of[p2s] = np.arange(len(p2s))
    fc = torch.zeros((len(p2s), N, 3, 3), dtype=torch.float64, device=device)
    for a, r in fc_rows_from_first_atoms(ops, atoms, disps, forces, N, device, row_of).items():
        if row_of[a] < 0:
            raise RuntimeError("fc2: displaced atom is not in p2s_map")
        fc[row_of[a]] = r
    map_atoms, map_ops = _sym_mappings(ops.perms, np.unique(atoms))
    distribute_rows(fc, p2s, map_atoms, map_ops, ops, row_of)
    fc2 = fc.cpu().numpy()
    if symmetrize:
        # translational + index-permutation symmetrisation of an (n_p, N, 3, 3)
        # array: phonopy's C routine, O(n_p N) on one host core.
        from phonopy.harmonic.force_constants import symmetrize_compact_force_constants

        symmetrize_compact_force_constants(fc2, prim)
    return fc2


# ---------------------------------------------------------------------------
# fc3
# ---------------------------------------------------------------------------
def _delta_fc2(ops, ds, i1, F1, F2, fc2_full, N, device, chunk=None):
    """_get_delta_fc2 for first displacement i1: constrained fc2 - fc2."""
    a1 = ds.first_atom[i1]
    G1 = ops.reduce_by_direction(ops.site_ops(a1), ds.first_disp[i1])
    s0, s1 = ds.sec_range[i1]
    sec_atoms = ds.sec_atom[s0:s1]
    sec_disps = ds.sec_disp[s0:s1]
    atom_list = np.unique(sec_atoms)

    # rows (a2 ascending, its second displacements in dataset order, bond ops in G1 order)
    perm_g = ops.perms[G1]  # (|G1|, N)
    order = np.argsort(sec_atoms, kind="stable")
    bond = perm_g[:, sec_atoms[order]].T == sec_atoms[order][:, None]  # (n_sec, |G1|)
    r_s, r_k = np.nonzero(bond)
    rows_f = order[r_s]
    rows_op = G1[r_k]
    rows_u = sec_disps[rows_f]
    rows_b = np.searchsorted(atom_list, sec_atoms[rows_f])
    nb = len(atom_list)
    counts = np.bincount(rows_b, minlength=nb)
    M = counts.max()
    pos = np.arange(len(rows_b)) - np.repeat(np.cumsum(counts) - counts, counts)
    op_idx = np.zeros((nb, M), dtype="int64")
    f_idx = np.zeros((nb, M), dtype="int64")
    u = np.zeros((nb, M, 3))
    valid = np.zeros((nb, M))
    op_idx[rows_b, pos] = rows_op
    f_idx[rows_b, pos] = rows_f
    u[rows_b, pos] = rows_u
    valid[rows_b, pos] = 1.0

    dF = F2 - F1[None]  # (n_sec, N, 3)
    fc = torch.zeros((N, N, 3, 3), dtype=torch.float64, device=device)
    step = chunk or max(1, int(BLOCK_BYTES // (M * N * 3 * 8 * 3)))
    for c0 in range(0, nb, step):
        sl = slice(c0, c0 + step)
        fc[torch.as_tensor(atom_list[sl], device=device)] = solve_rows(
            ops, torch.as_tensor(u[sl], device=device), dF,
            torch.as_tensor(op_idx[sl], device=device), torch.as_tensor(f_idx[sl], device=device),
            torch.as_tensor(valid[sl], device=device))
    # distribute_force_constants with the reduced site symmetry about a1
    map_a, map_k = _sym_mappings(perm_g, atom_list)
    distribute_rows(fc, np.arange(N), map_a, G1[map_k], ops, np.arange(N))
    return fc - fc2_full


def _solve_fc3_rows(ops, a1, disps_first, delta, N, device, i_chunk=None):
    """_solve_fc3: fc3[a1, i, j] = pinv(U) @ [R_k delta_d[inv_k i, inv_k j] R_k^T]."""
    S1 = ops.site_ops(a1)
    nd, ns = len(disps_first), len(S1)
    rc = ops.t_rc[torch.as_tensor(S1, device=device)]  # (ns, 3, 3)
    U = torch.einsum("sij,dj->dsi", rc, torch.as_tensor(disps_first, device=device)).reshape(-1, 3)
    invU = torch.linalg.pinv(U).reshape(3, nd, ns)  # (3, nd, ns)
    inv = ops.t_invperms[torch.as_tensor(S1, device=device)]  # (ns, N)
    out = torch.empty((N, N, 3, 3, 3), dtype=torch.float64, device=device)
    step = i_chunk or max(1, int(BLOCK_BYTES // (ns * N * 9 * 8 * 3)))
    for i0 in range(0, N, step):
        ii = inv[:, i0:i0 + step]  # (ns, I)
        acc = torch.zeros((ii.shape[1], N, 3, 3, 3), dtype=torch.float64, device=device)
        for d in range(nd):
            src = delta[d][ii[:, :, None], inv[:, None, :]]  # (ns, I, N, 3, 3)
            rot = torch.einsum("slm,sijmn,sqn->sijlq", rc, src, rc)
            acc += torch.einsum("as,sijlq->ijalq", invU[:, d], rot)
        out[i0:i0 + step] = acc
    return out


def _distribute_fc3_maps(ops, done, targets):
    """distribute_fc3 operation choice: for each target the first done atom (in order)
    reachable by some operation, preferring a pure translation among those operations.
    Returns {target: (done_atom, op)}."""
    choice = {}
    for t in targets:
        for d in done:
            ks = np.nonzero(ops.perms[:, t] == d)[0]
            if len(ks):
                pt = ks[ops.pure_trans[ks]]
                choice[t] = (d, pt[0] if len(pt) else ks[0])
                break
        else:
            raise RuntimeError("Position or symmetry may be wrong.")
    return choice


def _composed_rows(ops, first_rows_atoms, p2s, cutoff, N):
    """For every atom x: (least row f_x, atom map m_x, Cartesian C_x) with
    fc3[x, y, z] = C_x (x) C_x (x) C_x . F[f_x, m_x[y], m_x[z]]  (C = R^T of the op)."""
    first = np.unique(first_rows_atoms)
    f_of = {a: i for i, a in enumerate(first)}
    comp = {a: (f_of[a], np.arange(N), np.eye(3)) for a in first}
    stage1 = [i for i in p2s if i not in first]
    for t, (d, k) in _distribute_fc3_maps(ops, first, stage1).items():
        comp[t] = (f_of[d], ops.perms[k], ops.rc[k].T)
    if cutoff:
        done = np.unique(np.concatenate((first, p2s)))
        rest = [i for i in range(N) if i not in set(done)]
        for t, (d, k) in _distribute_fc3_maps(ops, done, rest).items():
            fd, md, Cd = comp[d]
            # fc3[t,y,z] = C2 fc3[d, m2 y, m2 z] = C2 C1 F[fd, md[m2 y], md[m2 z]]
            comp[t] = (fd, md[ops.perms[k]], ops.rc[k].T @ Cd)
    return comp


def _fc3_done(ops, ds, supercell):
    """_get_fc3_done (as (N, N): fc3_done[a, b, :] is set as a whole)."""
    N = len(supercell)
    L = supercell.cell.T
    done = np.zeros((N, N), dtype=bool)
    for i1 in range(ds.n_first):
        a1 = ds.first_atom[i1]
        G1 = ops.reduce_by_direction(ops.site_ops(a1), ds.first_disp[i1])
        s0, s1 = ds.sec_range[i1]
        incl = ds.sec_incl[s0:s1]
        if all(v is not None for v in incl):
            least = ds.sec_atom[s0:s1][np.array(incl, dtype=bool)]
        else:
            least = _least_by_distance(ds, i1, a1, supercell, ops, L)
        least = np.unique(np.array(least, dtype="int64"))
        # phono3py keeps only the mapping of the last reduced rotation
        second = np.unique(ops.perms[G1[-1]][least]) if len(least) else least
        rot_a1 = ops.perms[:, a1]  # (n_ops,)
        if len(second):
            done[rot_a1[:, None], ops.perms[:, second]] = True
    return done


def _least_by_distance(ds, i1, a1, supercell, ops, L):
    """Second atoms within the cut-off when the dataset has no 'included' flags (slow path)."""
    from phono3py.phonon3.displacement_fc3 import get_smallest_vector_of_atom_pair

    least = []
    for y in ds.second_atoms[i1]:
        if "included" in y:
            if y["included"]:
                least.append(y["number"])
        elif ds.cutoff is not None:
            v = get_smallest_vector_of_atom_pair(a1, y["number"], supercell, ops.symprec)
            if np.linalg.norm(np.dot(L, v)) < ds.cutoff:
                least.append(y["number"])
    return least


def produce_fc3_compact(ph3, fc3_mmap, device, fast_sym_compat=True, progress=True):
    """produce_fc3(is_compact_fc=True) from the fc3 force mmap -> (n_p, N, N, 3, 3, 3)."""
    cell = ph3.supercell
    prim = ph3.primitive
    N = len(cell)
    ops = _Ops(cell, ph3.symmetry, device)
    ds = FCDataset(ph3.dataset)
    p2s = np.array(prim.p2s_map, dtype="int64")
    t0 = time.time()

    # full fc2 of the fc3 supercell from the first displacements (delta reference)
    F1_all = _load_forces(fc3_mmap, (0, ds.n_first), device)
    rows = fc_rows_from_first_atoms(ops, ds.first_atom, ds.first_disp, F1_all, N, device, None)
    fc2_full = torch.zeros((N, N, 3, 3), dtype=torch.float64, device=device)
    done = np.array(sorted(rows))
    for a, r in rows.items():
        fc2_full[a] = r
    map_a, map_k = _sym_mappings(ops.perms, done)
    distribute_rows(fc2_full, np.arange(N), map_a, map_k, ops, np.arange(N))

    # least-atom rows
    first_atoms = np.unique(ds.first_atom)
    F = torch.empty((len(first_atoms), N, N, 3, 3, 3), dtype=torch.float64, device=device)
    for ia, a1 in enumerate(first_atoms):
        sel = np.nonzero(ds.first_atom == a1)[0]
        deltas = []
        for i1 in sel:
            F2 = _load_forces(fc3_mmap, ds.force_index_second(i1), device)
            deltas.append(_delta_fc2(ops, ds, i1, F1_all[i1], F2, fc2_full, N, device))
            del F2
        F[ia] = _solve_fc3_rows(ops, a1, ds.first_disp[sel], torch.stack(deltas), N, device)
        del deltas
        if progress:
            log(f"[FC3-GPU] first atom {ia + 1}/{len(first_atoms)} ({time.time() - t0:.1f} s)")

    comp = _composed_rows(ops, first_atoms, p2s, ds.cutoff is not None, N)
    fc3 = _assemble_compact(F, comp, p2s, N, device,
                            done=_fc3_done(ops, ds, cell) if ds.cutoff is not None else None)
    if fast_sym_compat:
        _pfrun_fast_sym_compat(fc3)
    return fc3.cpu().numpy()


class _Composed:
    """Tensors of _composed_rows: f (N,), m (N, N), C (N, 3, 3)."""

    def __init__(self, comp, N, device):
        # atoms without an entry (no cut-off: only p2s rows are needed) get a dummy
        dummy = (0, np.arange(N), np.eye(3))
        rows = [comp.get(x, dummy) for x in range(N)]
        self.f = torch.as_tensor(np.array([r[0] for r in rows]), device=device)
        self.m = torch.as_tensor(np.stack([r[1] for r in rows]), device=device)
        self.C = torch.as_tensor(np.stack([r[2] for r in rows]), device=device)


def _rot3(C, src):
    """C (x) C (x) C applied to the last three axes; C (..., 3, 3) broadcast over src."""
    src = torch.einsum("...ai,...ijk->...ajk", C, src)
    src = torch.einsum("...bj,...ajk->...abk", C, src)
    return torch.einsum("...ck,...abk->...abc", C, src)


def _row_full(F, cp, x):
    """fc3[x, :, :] -> (N, N, 3, 3, 3)."""
    m = cp.m[x]
    return _rot3(cp.C[x], F[cp.f[x]][m[:, None], m[None, :]])


def _col_p(F, cp, p, second):
    """X[x, y] = fc3[x, y, p] (second=False) or fc3[x, p, y] (second=True), all x, y."""
    mp = cp.m[:, p]  # (N,) image of p under each x's map
    if second:
        src = F[cp.f[:, None], mp[:, None], cp.m]
    else:
        src = F[cp.f[:, None], cp.m, mp[:, None]]
    return _rot3(cp.C[:, None], src)


def _assemble_compact(F, comp, p2s, N, device, done=None):
    """Compact rows fc3[p2s].  With cut-off: _cutoff_fc3_for_cutoff_pairs averaging,
    evaluated directly for (a, b, c) = (p, j, k)."""
    cp = _Composed(comp, N, device)
    n_p = len(p2s)
    out = torch.empty((n_p, N, N, 3, 3, 3), dtype=torch.float64, device=device)
    if done is None:
        for ip, p in enumerate(p2s):
            out[ip] = _row_full(F, cp, int(p))
        return out
    D = torch.as_tensor(done, device=device, dtype=torch.float64)
    for ip, p in enumerate(p2s):
        p = int(p)
        T = _row_full(F, cp, p)          # [j, k] fc3[p, j, k]
        X1 = _col_p(F, cp, p, False)     # [x, y] fc3[x, y, p]
        X2 = _col_p(F, cp, p, True)      # [x, y] fc3[x, p, y]
        Dp = D[p]                        # D[p, :]
        Dc = D[:, p]                     # D[:, p]
        w = [Dp[:, None].expand(N, N),   # D[p, j]
             Dc[None, :].expand(N, N),   # D[k, p]
             D,                          # D[j, k]
             Dp[None, :].expand(N, N),   # D[p, k]
             Dc[:, None].expand(N, N),   # D[j, p]
             D.T]                        # D[k, j]
        terms = [T,                                        # fc3[p,j,k][l,m,n]
                 X2.transpose(0, 1).permute(0, 1, 3, 4, 2),  # fc3[k,p,j][n,l,m]
                 X1.permute(0, 1, 4, 2, 3),                # fc3[j,k,p][m,n,l]
                 T.transpose(0, 1).permute(0, 1, 2, 4, 3),   # fc3[p,k,j][l,n,m]
                 X2.permute(0, 1, 3, 2, 4),                # fc3[j,p,k][m,l,n]
                 X1.transpose(0, 1).permute(0, 1, 4, 3, 2)]  # fc3[k,j,p][n,m,l]
        acc = sum(t * wi[..., None, None, None] for t, wi in zip(terms, w))
        s = sum(w)
        ok = s > 0
        out[ip] = torch.where(ok[..., None, None, None],
                              acc / torch.where(ok, s, torch.ones_like(s))[..., None, None, None],
                              torch.zeros_like(acc))
    return out


def _pfrun_fast_sym_compat(fc3):
    """pf-run's fast_symmetrize_fc3: _phono3py.permutation_symmetry_fc3 called on the
    *compact* array, i.e. with num_atom = n_p, so it symmetrises the first n_p^3 * 27
    numbers of the buffer as if they were an (n_p, n_p, n_p, 3, 3, 3) array.
    Reproduced for compatibility with fc3 files written by pf-run."""
    n = fc3.shape[0]
    flat = fc3.reshape(-1)[: n ** 3 * 27].view(n, n, n, 3, 3, 3)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    s = sum(flat.permute(*a, *(3 + i for i in a)) for a in perms) / 6
    flat.copy_(s)
