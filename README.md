# differentiable-tebd

A Python package using mostly JAX to implement the time-evolving block decimation algorithm in a differentiable way along with some data generation tools to synthesize Hamiltonian learning problems.

This package was used to generate the numerical results in our paper on [Scalably learning quantum many-body Hamiltonians from dynamical data](https://arxiv.org/abs/2209.14328).

## Environment

The default type for complex-valued arrays is double precision, i.e. `jax.numpy.complex128`.
To work with single precision, set the environment variable `TEBD_COMPLEX_TYPE=complex64` before importing this package.

## Installation

Install the package via `pip install -e /path/to/differentiable-tebd`.
The only dependency that is not installed automatically is `jaxlib`.
You need to install it manually as this differs depending on the hardware you want to use ([installation guide](https://github.com/google/jax/#installation)).

## Versioning

Be sure to checkout the relevant or latest version tag (e.g. `git checkout 0.0.0`) to be able to reproduce results easily.

## Demo

For a brief demo for how to use this package please have a look at our [repository](https://github.com/frederikwilde/scalable-dynamical-hamiltonian-learning/tree/main/demo) for the paper.

## Citing

If you use (parts of) this package, please cite our [paper](https://arxiv.org/abs/2209.14328).
```
@misc{wilde_scalably_2022,
  doi = {10.48550/ARXIV.2209.14328},
  url = {https://arxiv.org/abs/2209.14328},
  author = {Wilde, Frederik and Kshetrimayum, Augustine and Roth, Ingo and Hangleiter, Dominik and Sweke, Ryan and Eisert, Jens},
  title = {Scalably learning quantum many-body Hamiltonians from dynamical data},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
