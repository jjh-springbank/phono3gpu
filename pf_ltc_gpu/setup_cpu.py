# -*- coding: utf-8 -*-
"""CPU-side setup: structure, force constants, BZ grid, triplets.

Only light bookkeeping happens here (file I/O, symmetry, grid indices).
Everything heavy (phonons, ph-ph interaction, integration weights, kappa)
lives in the torch kernels.
"""
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phono3py.other.tetrahedron_method import get_tetrahedra_relative_grid_address
from phono3py.phonon.grid import get_grid_points_by_rotations, get_ir_grid_points
from phono3py.phonon3.triplets import get_triplets_at_q
from phonopy import units as U


def log(msg):
    print(msg, flush=True)


def load_config(path):
    spec = importlib.util.spec_from_file_location("ltc_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LTC_CONFIG


# ---------------------------------------------------------------------------
# Mesh selection: copied from pf_ltc/runner.py (GRG fallback for low-symmetry
# cells, see pf-ltc-grg-mesh-bug).
# ---------------------------------------------------------------------------
def calculate_auto_mesh(ph3_obj, density):
    lengths = np.linalg.norm(ph3_obj.unitcell.cell, axis=1)
    return [max(1, int(round(density / l))) for l in lengths], lengths


def set_mesh_numbers_with_grg_fallback(ph3_obj, mesh):
    try:
        ph3_obj.mesh_numbers = mesh
        return False
    except RuntimeError as e:
        msg = str(e)
        if "Grid symmetry is broken" not in msg:
            raise
        log(f"[Mesh] Regular grid failed: {msg}")
        log("[Mesh] Retrying with generalized regular grid (GRG) ...")

    import phono3py.api_phono3py as api

    _orig_BZGrid = api.BZGrid

    def _BZGrid_force_grg(*args, **kwargs):
        kwargs["use_grg"] = True
        kwargs["force_SNF"] = True
        log(f"[Mesh][DBG] BZGrid wrapper called. use_grg={kwargs.get('use_grg')} "
            f"force_SNF={kwargs.get('force_SNF')}")
        return _orig_BZGrid(*args, **kwargs)

    if hasattr(ph3_obj, "_bz_grid"):
        ph3_obj._bz_grid = None
    api.BZGrid = _BZGrid_force_grg
    try:
        ph3_obj.mesh_numbers = mesh
    finally:
        api.BZGrid = _orig_BZGrid
    log("[Mesh] GRG forced successfully.")
    return True


def try_mesh_candidates(ph3, mesh0):
    a, b, c = [int(x) for x in mesh0]
    candidates = [
        [a, b, c],
        [a + 1, b + 1, c],
        [a + 1, b + 1, c + 1],
        [a + 2, b + 2, c],
        [a + 2, b + 2, c + 2],
        [max(1, a - 1), max(1, b - 1), max(1, c - 1)],
        [16, 16, 8],
        [15, 15, 9],
        [16, 16, 10],
        [18, 18, 10],
        [12, 12, 6],
    ]
    tried = set()
    last_err = None
    for m in candidates:
        m = [int(x) for x in m]
        if tuple(m) in tried:
            continue
        tried.add(tuple(m))
        log(f"[Mesh] Trying mesh={m} (regular->GRG)")
        try:
            used_grg = set_mesh_numbers_with_grg_fallback(ph3, m)
            log(f"[Mesh] OK mesh={m} | GRG={used_grg}")
            return m, used_grg
        except Exception as e:
            last_err = e
            log(f"[Mesh] FAIL mesh={m}: {e}")
    raise RuntimeError(f"[Mesh] All mesh candidates failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
def _grid_index(addr, D):
    """recgrid_get_grid_index_from_address: address mod D -> dense index."""
    a = np.mod(addr, D)
    return a[..., 0] + D[0] * a[..., 1] + D[0] * D[1] * a[..., 2]


def neighbor_table(bz_grid, rel_addr):
    """BZ-grid points of addresses[gp] + rel_addr for every BZ grid point.

    Port of tpi_get_neighboring_grid_points (type-2 dense gp_map).
    rel_addr : (..., 3) int.  Returns (n_bz, ...) int64.
    """
    if not bz_grid.store_dense_gp_map:
        raise NotImplementedError("Only dense gp_map (type 2) BZ grids are supported.")
    D = np.array(bz_grid.D_diag, dtype="int64")
    adrs = np.array(bz_grid.addresses, dtype="int64")
    gp_map = np.array(bz_grid.gp_map, dtype="int64")
    shape = rel_addr.shape[:-1]
    rel = rel_addr.reshape(-1, 3)
    bz_addr = adrs[:, None, :] + rel[None, :, :]  # (n_bz, n_rel, 3)
    gp = _grid_index(bz_addr, D)
    out = gp_map[gp].copy()
    start = gp_map[gp]
    n_cand = gp_map[gp + 1] - start
    multi = n_cand > 1
    if multi.any():
        idx = np.nonzero(multi)
        found = np.zeros(len(idx[0]), dtype=bool)
        for m in range(int(n_cand.max())):
            cand = start[idx] + m
            ok = (m < n_cand[idx]) & ~found
            cand_c = np.where(ok, cand, 0)
            match = ok & (adrs[cand_c] == bz_addr[idx]).all(axis=1)
            out[idx[0][match], idx[1][match]] = cand[match]
            found |= match
    return out.reshape((len(adrs),) + shape)


@dataclass
class LTCSetup:
    ph3: object
    conf: dict
    mesh: list
    used_grg: bool
    temperatures: np.ndarray
    cutoff_frequency: float
    factor: float  # VaspToTHz
    # fc2 / dynamical matrix data (phonon_primitive / phonon_supercell)
    fc2: np.ndarray
    dm_svecs: np.ndarray
    dm_multi: np.ndarray
    dm_masses: np.ndarray
    dm_p2s: np.ndarray
    dm_s2pp: np.ndarray  # supercell atom -> primitive atom index
    # fc3 data (primitive / supercell)
    fc3: np.ndarray
    svecs: np.ndarray
    multi: np.ndarray
    masses: np.ndarray
    p2s: np.ndarray
    s2pp: np.ndarray
    all_shortest: np.ndarray
    make_r0_average: bool
    pp_unit_conversion: float
    ise_unit_conversion: float
    # grid
    bz_grid: object
    ir_grid_points: np.ndarray
    grid_weights: np.ndarray
    qpoints_bz: np.ndarray  # (n_bz, 3) reduced coords of all BZ grid points
    nb_plus: np.ndarray  # (n_bz, 24, 4) tetrahedron vertices, +rel
    nb_minus: np.ndarray  # (n_bz, 24, 4) tetrahedron vertices, -rel
    rotations_cartesian: np.ndarray
    lattice: np.ndarray  # primitive cell (rows)
    volume: float
    mass_variances: np.ndarray = None
    extra: dict = field(default_factory=dict)


def _s2pp(primitive):
    p2p = primitive.p2p_map
    return np.array([p2p[i] for i in primitive.s2p_map], dtype="int64")


def build_setup(conf, mesh_override=None, fc_builder=None):
    """Load structure + fc2/fc3 and prepare all grid bookkeeping on CPU.

    fc_builder(kind, ph3) -> compact fc array is called when the fc hdf5 file does
    not exist (pf-run computes it from the force mmap in that case)."""
    load_kwargs = {"produce_fc": False, "log_level": 0}
    if conf.get("symprec") is not None:
        load_kwargs["symprec"] = conf["symprec"]
    # phonopy >= 3 defaults primitive_matrix to 'auto' (it was the identity before). Keep the
    # identity when the yaml has no primitive_matrix, so every phono3py version sees the same cell.
    if conf.get("primitive_matrix") is not None:
        load_kwargs["primitive_matrix"] = conf["primitive_matrix"]
    elif "primitive_matrix" not in Path(conf["phono3py_yaml"]).read_text():
        load_kwargs["primitive_matrix"] = "P"
    ph3 = phono3py.load(conf["phono3py_yaml"], **load_kwargs)

    if fc_builder is not None and not Path(conf["fc2_hdf5"]).exists():
        fc2 = fc_builder("fc2", ph3)
    else:
        log(f"[FC2] Loading {conf['fc2_hdf5']}...")
        fc2 = read_fc2_from_hdf5(filename=conf["fc2_hdf5"], p2s_map=ph3.phonon_primitive.p2s_map)
    if fc_builder is not None and not Path(conf["fc3_hdf5"]).exists():
        fc3 = fc_builder("fc3", ph3)
    else:
        log(f"[FC3] Loading {conf['fc3_hdf5']}...")
        fc3 = read_fc3_from_hdf5(filename=conf["fc3_hdf5"], p2s_map=ph3.primitive.p2s_map)
    n_p = len(ph3.primitive)
    if fc3.shape[0] != n_p:
        raise RuntimeError(f"fc3 must be compact (got shape {fc3.shape}).")
    if fc2.shape[0] != len(ph3.phonon_primitive):
        raise RuntimeError(f"fc2 must be compact (got shape {fc2.shape}).")
    if conf.get("rotational_sum_rule"):  # optional correction, off by default (see rotsum.py)
        from .rotsum import apply_from_config
        fc2, fc3 = apply_from_config(conf, ph3, fc2, fc3, log=log)
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    if conf.get("stop_after_force_constants", False):
        return None

    mesh = mesh_override if mesh_override is not None else conf.get("mesh", None)
    if mesh is None:
        mesh, _ = calculate_auto_mesh(ph3, conf.get("mesh_density", 50.0))
        log(f"[AutoMesh] {mesh}")
    mesh, used_grg = try_mesh_candidates(ph3, mesh)
    log(f"[Mesh] FINAL mesh_numbers={mesh} | GRG={used_grg}")

    bz_grid = ph3.grid
    factor = U.VaspToTHz

    # Same unit constants as Interaction / ImagSelfEnergy.
    Hbar, EV, Angstrom, THz, AMU = U.Hbar, U.EV, U.Angstrom, U.THz, U.AMU
    num_grid = int(np.prod(bz_grid.D_diag))
    pp_unit = ((Hbar * EV) ** 3 / 36 / 8 * EV ** 2 / Angstrom ** 6
               / (2 * np.pi * THz) ** 3 / AMU ** 3 / num_grid / EV ** 2)
    ise_unit = 18 * np.pi / (Hbar * EV) ** 2 / (2 * np.pi * THz) ** 2 * EV ** 2

    # fc3-side atom correspondence (Interaction uses ph3.primitive)
    prim = ph3.primitive
    svecs, multi = prim.get_smallest_vectors()
    all_shortest = _get_all_shortest(prim, svecs, multi, ph3.primitive_symmetry.tolerance)

    # fc2-side (dynamical matrix uses phonon_primitive / phonon_supercell)
    pprim = ph3.phonon_primitive
    dm_svecs, dm_multi = pprim.get_smallest_vectors()

    ir_gps, ir_weights, _ = get_ir_grid_points(bz_grid)
    ir_gps = np.array(bz_grid.grg2bzg[ir_gps], dtype="int64")

    qpoints_bz = np.dot(bz_grid.addresses, bz_grid.QDinv.T)

    rel = np.array(
        np.dot(get_tetrahedra_relative_grid_address(bz_grid.microzone_lattice), bz_grid.P.T),
        dtype="int64", order="C")  # (24, 4, 3)
    nb_plus = neighbor_table(bz_grid, rel)
    nb_minus = neighbor_table(bz_grid, -rel)

    temps = np.array(conf.get("temperatures", [300.0, 500.0, 800.0]), dtype="double")

    s = LTCSetup(
        ph3=ph3, conf=conf, mesh=mesh, used_grg=used_grg, temperatures=temps,
        cutoff_frequency=float(ph3._cutoff_frequency), factor=factor,
        fc2=np.ascontiguousarray(fc2, dtype="double"),
        dm_svecs=np.array(dm_svecs, dtype="double"),
        dm_multi=np.array(dm_multi, dtype="int64"),
        dm_masses=np.array(pprim.masses, dtype="double"),
        dm_p2s=np.array(pprim.p2s_map, dtype="int64"),
        dm_s2pp=_s2pp(pprim),
        fc3=np.ascontiguousarray(fc3, dtype="double"),
        svecs=np.array(svecs, dtype="double"),
        multi=np.array(multi, dtype="int64"),
        masses=np.array(prim.masses, dtype="double"),
        p2s=np.array(prim.p2s_map, dtype="int64"),
        s2pp=_s2pp(prim),
        all_shortest=all_shortest,
        make_r0_average=bool(ph3._make_r0_average),
        pp_unit_conversion=pp_unit,
        ise_unit_conversion=ise_unit,
        bz_grid=bz_grid,
        ir_grid_points=ir_gps,
        grid_weights=np.array(ir_weights, dtype="int64"),
        qpoints_bz=qpoints_bz,
        nb_plus=nb_plus,
        nb_minus=nb_minus,
        rotations_cartesian=np.array(bz_grid.rotations_cartesian, dtype="double"),
        lattice=np.array(prim.cell, dtype="double"),
        volume=float(prim.volume),
    )
    if ph3.nac_params is not None and conf.get("is_nac", True):
        s.extra["nac"] = gonze_nac_dataset(ph3, fc2)
        log(f"[NAC] Gonze-Lee: {len(s.extra['nac']['G_list'])} G vectors, "
            f"Lambda={s.extra['nac']['Lambda']:.4f}")
    return s


def gonze_nac_dataset(ph3, fc2):
    """Gonze-Lee NAC inputs (host, once): short-range fc2, dd_q0, G list, Lambda.

    Uses phonopy's DynamicalMatrixGL exactly as phono3py's Interaction does; the
    per-q reciprocal dipole-dipole sum is then evaluated on the device (DynMat).
    """
    from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL
    nac = ph3.nac_params
    dm = DynamicalMatrixGL(ph3.phonon_supercell, ph3.phonon_primitive, fc2, nac_params=nac)
    dm.make_Gonze_nac_dataset()
    fc_sr, dd_q0, _g_cutoff, g_list, lam = dm.Gonze_nac_dataset
    pprim = ph3.phonon_primitive
    return dict(fc_short_range=np.ascontiguousarray(fc_sr, dtype="double"),
                dd_q0=np.array(dd_q0, dtype="complex128").reshape(len(pprim), 3, 3),
                G_list=np.array(g_list, dtype="double"), Lambda=float(lam),
                born=np.array(nac["born"], dtype="double"),
                dielectric=np.array(nac["dielectric"], dtype="double"),
                factor=float(nac["factor"]) * 4 * np.pi / pprim.volume,
                positions=np.array(pprim.positions, dtype="double"),
                rec_lattice=np.linalg.inv(pprim.cell))


def _get_all_shortest(primitive, svecs, multi, symprec):
    """Port of Interaction._get_all_shortest (vectorized over k)."""
    n_satom, n_patom, _ = multi.shape
    perms = primitive.atomic_permutations
    p2s = np.array(primitive.p2s_map, dtype="int64")
    s2pp = _s2pp(primitive)
    lattice = primitive.cell
    out = np.zeros((n_patom, n_satom, n_satom), dtype="byte")
    single_k = multi[:, :, 0] == 1  # (n_satom, n_patom)
    for i_patom in range(n_patom):
        for j_atom in range(n_satom):
            if multi[j_atom, i_patom, 0] > 1:
                continue
            j_patom = s2pp[j_atom]
            i_perm = np.where(perms[:, j_atom] == p2s[j_patom])[0]
            assert len(i_perm) == 1
            k_mapped = perms[i_perm[0], :]  # (n_satom,)
            ok = single_k[:, i_patom] & single_k[k_mapped, j_patom]
            vec_jk = svecs[multi[:, i_patom, 1]] - svecs[multi[j_atom, i_patom, 1]]
            d_jk = np.linalg.norm(vec_jk @ lattice, axis=1)
            d_jk_mapped = np.linalg.norm(svecs[multi[k_mapped, j_patom, 1]] @ lattice, axis=1)
            out[i_patom, j_atom, ok & (np.abs(d_jk_mapped - d_jk) < symprec)] = 1
    return out


def triplets_at_gp(s: LTCSetup, gp):
    """Irreducible triplets (BZ indices) and weights at BZ grid point gp."""
    triplets, weights, _, _ = get_triplets_at_q(gp, s.bz_grid)
    return np.array(triplets, dtype="int64"), np.array(weights, dtype="int64")


def kstar_info(s: LTCSetup, gp):
    """(n_rot // order_kstar, order_kstar) as in ConductivityWignerMixIn."""
    rotation_map = get_grid_points_by_rotations(gp, s.bz_grid)
    order_kstar = len(np.unique(rotation_map))
    return len(rotation_map) // order_kstar, order_kstar
