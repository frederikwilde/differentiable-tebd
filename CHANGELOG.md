# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- `mps_utils.probability` now requires the MPS in three different bases: all-X, all-Y, and all-Z.
Those can be generated with the new `local_basis_transform` function. With this the number of basis
transformations is reduced to the constant `2 * num_sites`.

## [0.0.3] - 2021-05-03
### Added
- `draw_samples` accepts `cache_maxsize` as a keyword argument which controls the amount of caching
done for basis transformations.

## [0.0.2] - 2021-04-28
### Removed
- `generate_data_from_vecs` function is only useful for creating histograms. To enable subsampling
the bitstrings should be saved as they are.
### Added
- Keyword argument `seed` in `mps_utils.mps_zero_state` for reducibility and seperation from other
PRNG calls. If not set a fresh PRNG is created with a seed generated from the OS. Also the MPS
perturbations are now in the range `[-1, 1]` and `[-1j, 1j]` instead of `[0, 1]` and `[0, 1j]`.
### Changed
- Keyword argument `seed` is no longer accepted in `sampling.draw_samples`, instead a `np.random.Generator`
can be supplied under the `rng` keyword argument.

## [0.0.1] - 2021-03-20
### Added
New keyword argument `sequential_sampling` in `sampling.draw_samples()`.
When drawing `num_samples` samples from a state vector `vec` with the `draw_samples()`
function the underlying `scipy.stats` function allocates an array of size `(num_samples, len(vec))`.
(As a guideline, for `num_samples=10000` and `num_sites=24` this would amount to 156 GiB.)
This can be prohibitively large for large states or numbers of samples.
When `sequential_sampling` is set `True` the function now draws samples one by one, trading time for
memory.
This keyword argument can be set in the `generate_data_from_vecs` function.

## [0.0.0] - 2021-03-18
### Added
As of now the package consists of the following four modules:
- `mps_utils` containing functions for initializing and manipulating matrix product states.
- `sampling` containing functions to generate sythetic datasets of measurement outcomes.
- `state_vector_simulation` containing a state-vector simulator based on SciPy's `solve_ivp`
differential equation solver.
- `physical_models` containing functions for multiplying and time-evolving under the Hamiltonian
of specific models. Currently only the Heisenberg model with 4 free parameters is available.