# differentiable-tebd

A Python package using mostly JAX to implement the time-evolving block decimation algorithm in a differentiable way along with some data generation tools to synthesize Hamiltonian learning problems.

## Environment

The default type for complex-valued arrays is double precision, i.e. `jax.numpy.complex128`.
To work with single precision, set the environment variable `TEBD_COMPLEX_TYPE=complex64` before importing this package.

## Installation

Install the package via `pip install -e /path/to/differentiable-tebd`.
The only dependency that is not installed automatically is `jaxlib`.
You need to install it manually as this differs depending on the hardware you want to use ([installation guide](https://github.com/google/jax/#installation)).

## Versioning

Be sure to checkout the latest version tag (e.g. `git checkout 0.0.0`) to be able to reproduce results easily.
