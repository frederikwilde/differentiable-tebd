# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- The `sampling` module is now relocated to `sampling.qubits` to allow for future integration of
non-qubit systems.

## [0.2.4] - 2022-09-29
### Changed
- Refactoring. Moved the Trotter-step functions into the `utils.tebd` module. MPS methods specific
to physical dimension 2 are separated, in `mps_qubits`, from generic MPS methods.
- `utils.tebd.trotter_step` and `utils.tebd.trotter_step_order2` accept gates in the bulk as one
gate, which gets applied to all specified sites, or an array gates to be applied to the specified
sites.
### Added
- Tests for the gradient of Heisenberg and disordered Heisenberg model.

## [0.2.3] - 2022-06-05
### Changed
- Fixed bug in SVD JVP rule.

## [0.2.2] - 2022-06-03
### Changed
- Replaced JAX' deprecated `index_update` function by the new notation.
- Implemented custom JVP rule for the SVD to avoid issues with degeneracies.

## [0.2.1] - 2022-04-12
### Changed
- `state_vector_simulation.vec_simulation` accepts a user specified `initial_state` now.
### Added
- The function `mps_utils.mps_neel_state` to prepare the Neel state as an MPS.

## [0.2.0] - 2022-01-11
### Changed
- `mps_utils.contract_and_split` and `mps_utils.apply_gate` now return the sum of all _squared_ errors.
I.e. all truncated singular values squared and summed up.
- `physical_models.heisenberg` and `physical_models.heisenberg_disordered` also return the squared
error now.

## [0.1.1] - 2021-12-08
### Changed
- `sampling.save_basis_transforms` now uses Pickle protocol 4 instead of 3 (which is the default).
This enables the storage of transforms beyond the size of 4GB.
### Added
- `sampling.draw_samples_from_mps` is implemented.

## [0.1.0] - 2021-07-09
### Added
- Disordered Heisenberg model which has uniform XX, YY, and ZZ interactions, but a varying local X field.
I.e. the number of parameters is `3 + num_sites`.

## [0.0.5] - 2021-07-02
### Added
- New function `sampling.save_basis_transforms` pickles all basis transforms to disk. This effectively
enables caching when the system size is very large.
- Test module. First tests are on the `sampling` module.
### Changed
- `sampling.draw_samples` accepts `basis_transforms_dir` as a keyword argument for reading the
the transformations from disk. The directory can be created an populated before with the new
`sampling.save_basis_transforms` method.
### Removed
- `cache_maxsize` keyword argument of `sampling.draw_samples` is ignored now, since the caching
function needs to be defined outside of `draw_samples`.

## [0.0.4] - 2021-05-17
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