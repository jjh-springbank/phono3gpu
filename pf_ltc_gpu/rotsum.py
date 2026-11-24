# -*- coding: utf-8 -*-
"""Optional correction of fc2 (and fc3) to obey the rotational sum rules / Born-Huang equilibrium conditions.

Off by default. Switched on with the config key
    LTC_CONFIG["rotational_sum_rule"] = "born-huang"   # or "born" (rotational rule only), "born-huang+fc3"
(pf-run-gpu ... --rotational-sum-rule MODE sets the same key), or studied with `pf-run-gpu analyze rotsum`.

fc2.  With R(i, s) the vector from primitive atom i to supercell atom s (mean over equivalent shortest images,
the vectors that enter the small-q expansion of the dynamical matrix), the corrected fc2 obeys
    translation   sum_s Phi_ab(i, s) = 0
    rotation      sum_s [Phi_ab(i, s) R_g(i, s) - Phi_ag(i, s) R_b(i, s)] = 0            (Born, zero forces)
    Huang         sum_is [Phi_ab(i, s) R_g R_d - Phi_gd(i, s) R_a R_b] = 0               (zero stress)
The correction is the smallest change (Frobenius norm of the compact array) that fulfils these linear conditions
and keeps the index-permutation symmetry Phi_ab(i, s) = Phi_ba(s, i): d = -C'^T (C' C'^T)^+ r, with C the
condition rows, C' their permutation-symmetric part and r the residual. The conditions and the norm are
invariant under the space group, so a symmetric fc2 stays symmetric (checked and reported).
NAC (Gonze-Lee): the conditions are imposed on the full analytic dynamical matrix near Gamma, i.e. short-range
fc2 plus the analytic part of the dipole-dipole term; the residual r is taken from the small-q expansion of that
matrix (finite differences), and the correction is added to the real-space fc2 (hence to the short-range part).
The non-analytic G = 0 term is not touched.

fc3 ("+fc3").  Rotational rule that links fc3 to fc2 (it makes the antisymmetric part of the Grueneisen tensor
vanish), for every pair (i, j) and a, b, [g, d]:
    sum_k [Phi_abg(i, j, k) R_d(i, k) - Phi_abd(i, j, k) R_g(i, k)]
        + Phi_gb(i, j) d_ad - Phi_db(i, j) d_ag + Phi_ag(i, j) d_bd - Phi_ad(i, j) d_bg = 0
(fc2 here on the fc3 supercell). Each (i, j) block is corrected with the smallest change, then the permutation
symmetry and the translational sum rule of fc3 are restored with the phono3py routines; the two steps are
repeated (alternating projections). Needs fc2 on the fc3 supercell: used directly when the supercells are the
same; when the fc2 supercell is a multiple of the fc3 supercell fc2 is folded onto it (the rule then holds exactly
only at q-points commensurate with the fc3 supercell); otherwise the fc3 step is skipped with a message.
"""
import time
from types import SimpleNamespace

import numpy as np

PAIRS = ((1, 2), (2, 0), (0, 1))
SYM6 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _log(msg):
    print(msg, flush=True)


def geometry(prim):
    """R (n_p, n_s, 3) and RR (n_p, n_s, 3, 3): mean shortest vector i -> s (Cartesian) and mean of its square."""
    svecs, multi = prim.get_smallest_vectors()
    lat = np.array(prim.cell)
    n_s, n_p = multi.shape[:2]
    R = np.zeros((n_p, n_s, 3))
    RR = np.zeros((n_p, n_s, 3, 3))
    for s in range(n_s):
        for i in range(n_p):
            m, a = multi[s, i]
            v = svecs[a:a + m] @ lat
            R[i, s] = v.mean(axis=0)
            RR[i, s] = np.einsum("ma,mb->ab", v, v) / m
    return R, RR


def pair_map(prim, supercell):
    """(ip, sp) with Phi_ab(i, s) = Phi_ba(ip[i, s], sp[i, s]) (index-permutation partner in the compact array)."""
    x = np.array(supercell.scaled_positions)
    p2s = np.array(prim.p2s_map)
    p2p = prim.p2p_map
    s2pp = np.array([p2p[j] for j in prim.s2p_map])
    n_p, n_s = len(p2s), len(x)
    ip = np.broadcast_to(s2pp[None, :], (n_p, n_s)).copy()
    sp = np.zeros((n_p, n_s), dtype="int64")
    t = x - x[p2s[s2pp]]  # lattice translation that brings the primitive image of s to s
    for i in range(n_p):
        tgt = x[p2s[i]][None, :] - t  # (n_s, 3)
        d = tgt[:, None, :] - x[None, :, :]
        d -= np.round(d)
        j = np.abs(d).sum(-1).argmin(axis=1)
        if np.abs(d[np.arange(n_s), j]).max() > 1e-5:
            raise RuntimeError("pair_map: translated atom not found")
        sp[i] = j
    return ip, sp


def perm_part(C, ip, sp):
    """Permutation-symmetric part of rows C (m, n_p, n_s, 3, 3)."""
    return 0.5 * (C + C[:, ip, sp].transpose(0, 1, 2, 4, 3))


def constraint_rows(R, RR, huang=True):
    """Rows (m, n_p, n_s, 3, 3) of the translation, rotation and Huang conditions, in this order."""
    n_p, n_s = R.shape[:2]
    rows = []
    for i in range(n_p):
        for a in range(3):
            for b in range(3):
                c = np.zeros((n_p, n_s, 3, 3)); c[i, :, a, b] = 1.0; rows.append(c)
    for i in range(n_p):
        for a in range(3):
            for b, g in PAIRS:
                c = np.zeros((n_p, n_s, 3, 3)); c[i, :, a, b] = R[i, :, g]; c[i, :, a, g] = -R[i, :, b]; rows.append(c)
    if huang:
        for x in range(6):
            for y in range(x + 1, 6):
                (a, b), (g, d) = SYM6[x], SYM6[y]
                c = np.zeros((n_p, n_s, 3, 3))
                c[:, :, a, b] += RR[:, :, g, d]
                c[:, :, g, d] -= RR[:, :, a, b]
                rows.append(c)
    return np.array(rows)


def _dynmat_cpu(ph3, fc2):
    """DynMat on the CPU exactly as build_setup makes it (with NAC when the case has it), masses, lattice."""
    from . import kernels as K
    from .setup_cpu import _s2pp, gonze_nac_dataset
    pprim = ph3.phonon_primitive
    sv, mu = pprim.get_smallest_vectors()
    sv, mu = np.array(sv, dtype="double"), np.array(mu, dtype="int64")
    masses = np.array(pprim.masses, dtype="double")
    nac = gonze_nac_dataset(ph3, fc2) if ph3.nac_params is not None else None
    dm = K.DynMat(nac["fc_short_range"] if nac else fc2, sv, mu, masses, _s2pp(pprim), "cpu", nac=nac)
    return dm, masses, np.array(pprim.cell, dtype="double")


def longwave_sums(ph3, fc2, h=2e-4):
    """Small-q expansion of the full analytic dynamical matrix, converted to real-space sums:
    S0[i, a, b] = sum_s Phi_ab, S1[i, a, b, g] = sum_s Phi_ab R_g, S2[a, b, g, d] = sum_is Phi_ab R_g R_d,
    and the first-order acoustic term P^T D1 P (largest element)."""
    import torch
    from .analysis.elastic import LongWave
    dm, masses, lat = _dynmat_cpu(ph3, fc2)
    lat_t = torch.as_tensor(lat)
    ctx = SimpleNamespace(dm=dm, device=torch.device("cpu"), s=SimpleNamespace(dm_masses=masses),
                          to_reduced=lambda qc: qc @ lat_t.T)
    lw = LongWave(ctx, h=h)
    n_p = len(masses)
    sm = np.sqrt(masses).repeat(3)
    mm = sm[:, None] * sm[None, :]
    eye = torch.eye(3, dtype=torch.float64)
    D1, D2 = [], {}
    D0 = None
    for g in range(3):
        D0, d1, d2 = lw.ray(eye[g], True)
        D1.append(d1.numpy()); D2[(g, g)] = d2.numpy()
    for g in range(3):
        for d in range(g + 1, 3):
            _, _, d2 = lw.ray(eye[g] + eye[d], True)
            D2[(g, d)] = D2[(d, g)] = (d2.numpy() - D2[(g, g)] - D2[(d, d)]) / 2
    blk = lambda M: (M * mm).reshape(n_p, 3, n_p, 3)
    S0 = blk(D0.numpy().real).sum(axis=2)  # (n_p, 3, 3)
    S1 = np.stack([blk(D1[g].imag / (2 * np.pi)).sum(axis=2) for g in range(3)], axis=-1)  # (n_p, 3, 3, 3)
    S2 = np.zeros((3, 3, 3, 3))
    for (g, d), M in D2.items():
        S2[:, :, g, d] = blk(-2 * M.real / (2 * np.pi) ** 2).sum(axis=(0, 2))
    P = lw.P.numpy()
    a1 = max(float(np.abs(P.T @ D1[g] @ P).max()) for g in range(3))
    return S0, S1, S2, a1


def residual_from_sums(S0, S1, S2, huang=True):
    r = list(S0.reshape(-1))
    n_p = S0.shape[0]
    for i in range(n_p):
        for a in range(3):
            for b, g in PAIRS:
                r.append(S1[i, a, b, g] - S1[i, a, g, b])
    if huang:
        for x in range(6):
            for y in range(x + 1, 6):
                (a, b), (g, d) = SYM6[x], SYM6[y]
                r.append(S2[a, b, g, d] - S2[g, d, a, b])
    return np.array(r)


def correct_fc2(ph3, fc2, huang=True, log=_log, h=2e-4):
    """Corrected compact fc2 (new array) and a report dict."""
    t0 = time.time()
    pprim, psc = ph3.phonon_primitive, ph3.phonon_supercell
    R, RR = geometry(pprim)
    n_p, n_s = R.shape[:2]
    if fc2.shape[:2] != (n_p, n_s):
        raise RuntimeError("correct_fc2 needs the compact fc2")
    ip, sp = pair_map(pprim, psc)
    C = constraint_rows(R, RR, huang)
    m = len(C)
    nac = ph3.nac_params is not None
    rep = dict(mode="born-huang" if huang else "born", nac=bool(nac), n_conditions=int(m))
    rep["permutation_asymmetry_before"] = float(np.abs(fc2 - fc2[ip, sp].transpose(0, 1, 3, 2)).max())
    r_real = C.reshape(m, -1) @ fc2.reshape(-1)
    S0, S1, S2, a1 = longwave_sums(ph3, fc2, h)
    r_lw = residual_from_sums(S0, S1, S2, huang)
    if not nac:  # the two routes must give the same residual (checks signs and factors of the expansion)
        rep["residual_longwave_vs_realspace_max_abs"] = float(np.abs(r_lw - r_real).max())
        r = r_real
    else:
        rep["residual_short_range_only_max_abs"] = float(np.abs(
            C.reshape(m, -1) @ np.asarray(_dynmat_cpu(ph3, fc2)[0].fc2.numpy()).reshape(-1)).max())
        r = r_lw
    nT, nR = 9 * n_p, 9 * n_p
    rep.update(residual_translation_before=float(np.abs(r[:nT]).max()),
               residual_rotation_before=float(np.abs(r[nT:nT + nR]).max()),
               residual_huang_before=float(np.abs(r[nT + nR:]).max()) if huang else 0.0,
               acoustic_first_order_before=a1)
    Cp = perm_part(C, ip, sp).reshape(m, -1)
    A = Cp @ Cp.T
    d = -(Cp.T @ (np.linalg.pinv(A, rcond=1e-10, hermitian=True) @ r)).reshape(fc2.shape)
    new = fc2 + d
    # ---- report
    S0, S1, S2, a1 = longwave_sums(ph3, new, h)
    r2 = residual_from_sums(S0, S1, S2, huang)
    rep.update(residual_translation_after=float(np.abs(r2[:nT]).max()),
               residual_rotation_after=float(np.abs(r2[nT:nT + nR]).max()),
               residual_huang_after=float(np.abs(r2[nT + nR:]).max()) if huang else 0.0,
               acoustic_first_order_after=a1,
               change_norm=float(np.linalg.norm(d)), fc2_norm=float(np.linalg.norm(fc2)),
               change_rel_norm=float(np.linalg.norm(d) / np.linalg.norm(fc2)),
               change_max_abs=float(np.abs(d).max()), fc2_max_abs=float(np.abs(fc2).max()),
               permutation_asymmetry_after=float(np.abs(new - new[ip, sp].transpose(0, 1, 3, 2)).max()))
    try:  # space-group symmetry of the change: compare with its symmetrised copy (phonopy routine)
        from phonopy.harmonic.force_constants import symmetrize_compact_force_constants
        ds = np.array(d, dtype="double", order="C")
        symmetrize_compact_force_constants(ds, pprim)
        rep["change_not_symmetric_part_max_abs"] = float(np.abs(ds - d).max())
    except Exception as e:  # the check is optional
        rep["change_not_symmetric_part_max_abs"] = f"not checked ({e})"
    rep["time_s"] = time.time() - t0
    log(f"[ROTSUM] fc2 {rep['mode']} (NAC={nac}): rotation residual {rep['residual_rotation_before']:.3e} -> "
        f"{rep['residual_rotation_after']:.3e}, Huang {rep['residual_huang_before']:.3e} -> "
        f"{rep['residual_huang_after']:.3e} eV/A; change |d|/|fc2| = {rep['change_rel_norm']:.3e}, "
        f"max |d| = {rep['change_max_abs']:.3e} eV/A^2 ({rep['time_s']:.1f} s)")
    return new, rep


# ---------------------------------------------------------------------------
# fc3
# ---------------------------------------------------------------------------
def fc3_residual(fc3, fc2s, R):
    """Residual (n_p, n_s, 3, 3, 3 pairs) of the fc3 rotational rule; fc2s = compact fc2 on the fc3 supercell."""
    n_p, n_s = fc3.shape[:2]
    out = np.zeros((n_p, n_s, 3, 3, 3))
    for i in range(n_p):
        t = np.einsum("jkabg,kd->jabgd", fc3[i], R[i])  # sum_k Phi_abg R_d
        for x, (g, d) in enumerate(PAIRS):
            v = t[..., g, d] - t[..., d, g]  # (n_s, 3, 3) over a, b
            f = fc2s[i]  # (n_s, 3, 3)
            # + Phi_gb d_ad - Phi_db d_ag + Phi_ag d_bd - Phi_ad d_bg
            v[:, d, :] += f[:, g, :]
            v[:, g, :] -= f[:, d, :]
            v[:, :, d] += f[:, :, g]
            v[:, :, g] -= f[:, :, d]
            out[i, :, :, :, x] = v
    return out


def correct_fc3(ph3, fc3, fc2s, n_iter=400, tol=1e-9, log=_log):
    """Alternating projections: block-wise smallest change for the rotational rule, then permutation symmetry and
    translational sum rule (phono3py routines). Returns (new fc3, report)."""
    from phono3py.phonon3.fc3 import set_permutation_symmetry_compact_fc3, set_translational_invariance_compact_fc3
    t0 = time.time()
    prim = ph3.primitive
    R, _ = geometry(prim)
    n_p, n_s = fc3.shape[:2]
    new = np.array(fc3, dtype="double", order="C")
    # per atom i the 3 x (n_s * 3) row blocks: residual_x = sum_{k, c} B[x, k, c] Phi_abc(i, j, k)
    proj = []
    for i in range(n_p):
        B = np.zeros((3, n_s, 3))
        for x, (g, d) in enumerate(PAIRS):
            B[x, :, g] = R[i, :, d]
            B[x, :, d] = -R[i, :, g]
        Bf = B.reshape(3, -1)
        proj.append((B, np.linalg.pinv(Bf @ Bf.T, hermitian=True)))
    res0 = fc3_residual(new, fc2s, R)
    rep = dict(residual_before=float(np.abs(res0).max()), fc3_norm=float(np.linalg.norm(fc3)), history=[])
    for it in range(n_iter):
        res = fc3_residual(new, fc2s, R)
        rmax = float(np.abs(res).max())
        rep["history"].append(rmax)
        if rmax < tol * max(1.0, rep["residual_before"]):
            break
        for i in range(n_p):
            B, Ainv = proj[i]
            lam = np.einsum("xy,jaby->jabx", Ainv, res[i])  # (n_s, 3, 3, 3)
            new[i] -= np.einsum("jabx,xkc->jkabc", lam, B)
        set_permutation_symmetry_compact_fc3(new, prim)
        set_translational_invariance_compact_fc3(new, prim)
    res = fc3_residual(new, fc2s, R)
    d = new - fc3
    rep.update(residual_after=float(np.abs(res).max()), iterations=len(rep["history"]),
               change_norm=float(np.linalg.norm(d)), change_rel_norm=float(np.linalg.norm(d) / np.linalg.norm(fc3)),
               change_max_abs=float(np.abs(d).max()), fc3_max_abs=float(np.abs(fc3).max()), time_s=time.time() - t0)
    log(f"[ROTSUM] fc3: residual {rep['residual_before']:.3e} -> {rep['residual_after']:.3e} eV/A^2 in "
        f"{rep['iterations']} iterations; change |d|/|fc3| = {rep['change_rel_norm']:.3e} ({rep['time_s']:.1f} s)")
    return new, rep


def fc2_on_fc3_supercell(ph3, fc2):
    """(fc2 as a compact array on the fc3 supercell, how). Same supercell: the array itself. fc2 supercell a
    multiple of the fc3 supercell: fc2 folded onto it (sum over the images of each atom); the fc3 rule then holds
    for the folded fc2, i.e. exactly only at q-points commensurate with the fc3 supercell. Otherwise (None, why)."""
    sc3, sc2 = ph3.supercell, ph3.phonon_supercell
    n_p = len(ph3.primitive)
    if fc2.shape[:2] == (n_p, len(sc3)) and np.allclose(np.array(sc2.cell), np.array(sc3.cell)):
        return fc2, "same supercell"
    if fc2.shape[0] != n_p:
        return None, "different primitive cells"
    L3 = np.array(sc3.cell)
    M = np.array(sc2.cell) @ np.linalg.inv(L3)
    if np.abs(M - np.round(M)).max() > 1e-5:
        return None, "the fc2 supercell is not a multiple of the fc3 supercell"
    x3 = np.array(sc3.scaled_positions)
    x2 = (np.array(sc2.scaled_positions) @ np.array(sc2.cell)) @ np.linalg.inv(L3)
    # same origin: primitive atom 0 of both supercells
    x2 = x2 - x2[ph3.phonon_primitive.p2s_map[0]] + x3[ph3.primitive.p2s_map[0]]
    d = x2[:, None, :] - x3[None, :, :]
    d -= np.round(d)
    j = np.abs(d).sum(-1).argmin(axis=1)
    if np.abs(d[np.arange(len(x2)), j]).max() > 1e-4:
        return None, "atoms of the fc2 supercell not found in the fc3 supercell"
    out = np.zeros((n_p, len(sc3), 3, 3))
    np.add.at(out, (slice(None), j), fc2)
    return out, "fc2 folded onto the fc3 supercell"

def apply_from_config(conf, ph3, fc2, fc3, log=_log):
    """Called by build_setup when conf['rotational_sum_rule'] is set. Returns (fc2, fc3) corrected."""
    mode = str(conf.get("rotational_sum_rule")).lower()
    if mode in ("true", "1", "yes"):
        mode = "born-huang"
    base = mode.replace("+fc3", "")
    if base not in ("born", "born-huang"):
        raise SystemExit(f"[Error] rotational_sum_rule = '{mode}': use 'born', 'born-huang' or add '+fc3'")
    fc2n, rep = correct_fc2(ph3, np.asarray(fc2, dtype="double"), huang=base == "born-huang", log=log)
    conf["_rotsum_report"] = {"fc2": rep}
    fc3n = fc3
    if mode.endswith("+fc3"):
        f2, how = fc2_on_fc3_supercell(ph3, fc2n)
        if f2 is None:
            log(f"[ROTSUM] fc3 not corrected: {how}")
        else:
            log(f"[ROTSUM] fc3 rule with {how}")
            fc3n, rep3 = correct_fc3(ph3, np.asarray(fc3, dtype="double"), f2, log=log)
            rep3["fc2_used"] = how
            conf["_rotsum_report"]["fc3"] = rep3
    return fc2n, fc3n
