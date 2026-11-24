# Third-party notice

Phono3GPU re-implements, on the GPU, algorithms of the phonopy and phono3py packages (dynamical matrix,
three-phonon interaction with the r0-average transform, linear tetrahedron integration weights, isotope
scattering, RTA and direct-solution conductivities) and of phono3py-wte (Wigner conductivity), and it calls the
phonopy/phono3py Python API to read inputs and to build grids, symmetry operations and triplets.

phonopy and phono3py are distributed under the BSD 3-Clause license, Copyright (c) Atsushi Togo and contributors:
https://github.com/phonopy/phonopy , https://github.com/phonopy/phono3py .
phono3py-wte: https://github.com/MSimoncelli/phono3py-wte .
