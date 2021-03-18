# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0] - 2021-03-18
### Added
As of now the package consists of the following four modules:
- `mps_utils` containing functions for initializing and manipulating matrix product states.
- `sampling` containing functions to generate sythetic datasets of measurement outcomes.
- `state_vector_simulation` containing a state-vector simulator based on SciPy's `solve_ivp`
differential equation solver.
- `physical_models` containing functions for multiplying and time-evolving under the Hamiltonian
of specific models. Currently only the Heisenberg model with 4 free parameters is available.