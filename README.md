# Phono3GPU

GPU-accelerated phonon thermal transport with linear tetrahedron integration.

Phono3GPU computes the lattice thermal conductivity of crystals from second- and third-order force constants on
a single GPU (PyTorch): relaxation-time approximation (RTA), direct solution of the linearised Boltzmann equation,
the Wigner (coherence) term and isotope scattering. It reads the same inputs as phono3py
(`phono3py_disp.yaml`, `fc2.hdf5`, `fc3.hdf5`) and writes the same `kappa-*.hdf5` output, and it is written to
reproduce the phono3py numbers (linear tetrahedron method by default, Gaussian smearing as an option).

This repository is research code that accompanies a manuscript in preparation
("Phono3GPU: GPU-Accelerated Linear Tetrahedron Integration for Phonon Thermal Transport").
The package and command names are still the working names (`pf_ltc_gpu`, `pf-run-gpu`).
Comments in the source that mention `pf-run` refer to the authors' CPU workflow built on phono3py, which this code replaces.

## What runs on the GPU

- phonon frequencies, eigenvectors and velocity operators on the whole mesh (batched eigensolver);
- third-order force constants in reciprocal space (r0-average transform) and their projection on normal modes
  (batched matrix products);
- linear tetrahedron integration weights for the three-phonon and isotope terms, written without branches and
  restricted to the entries that can be non-zero;
- linewidths, RTA and Wigner conductivities, and the collision matrix of the direct solution.

Reading the inputs, the grid, the symmetry operations and the triplets are prepared on the host with the
phonopy/phono3py Python API. Double precision by default; a mixed-precision mode is available.

## Analysis tools (`pf-run-gpu analyze ...`)

- `atom-scatter`: three-phonon and isotope linewidths resolved by atom (the per-atom parts add up to the linewidth),
  per-atom scattering of the heat carriers, optional split into scattering channels (`--channels FCUT`), mode weights
  on each atom, participation ratio; grid-point splitting (`--gp-part K/M`, `--gp-merge DIR`).
- Added on 2026-11-24 and still experimental (checked on a few test cases only): `gruneisen`, `qpoints`, `elastic`,
  `linewidth`, `attenuation`, `dos`, `tmatrix`, `rotsum`. See `pf-run-gpu analyze <name> --help`.

## Install

```bash
pip install -e .
```

Requirements: Python >= 3.9, PyTorch with CUDA, numpy, h5py, phono3py 3.12.1 (its Python API is used for the setup).

## Use

```bash
pf-run-gpu --config ltc_config.py --device cuda:0 --outdir out
pf-run-gpu analyze atom-scatter --config ltc_config.py --device cuda:0 --outdir out_atom
```

`ltc_config.py` defines a dictionary `LTC_CONFIG`, for example

```python
LTC_CONFIG = {
    "phono3py_yaml": "phono3py_disp.yaml",
    "fc2_hdf5": "fc2.hdf5", "fc3_hdf5": "fc3.hdf5",
    "mesh": [19, 19, 19],
    "temperatures": [300.0, 500.0, 800.0],
    "calc_mode": {"base_method": "RTA", "wigner_correction": True},   # or "LBTE"
    "is_isotope": True,
    "precision": "fp64",
    # rules of phono3py >= 4.8 for the tetrahedron weights (omit both for the 3.12.1 behaviour)
    "thm_vertex_limit": True,
    "average_degenerate_weights": True,
}
```

## Status and limits

- One GPU; the memory need grows with the number of bands (cells of a few hundred atoms fit on 48-96 GB cards
  in the RTA with mixed precision).
- Four-phonon scattering is not included.
- No automated test suite is shipped yet: the tests used during development depend on reference data that are
  not part of this repository.

## Acknowledgement and license

Phono3GPU follows the algorithms of [phonopy](https://github.com/phonopy/phonopy),
[phono3py](https://github.com/phonopy/phono3py) and [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte);
see `NOTICE.md`. Distributed under the BSD 3-Clause license (`LICENSE`).
