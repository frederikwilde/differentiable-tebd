import jax
import jax.numpy as jnp
from jax import custom_jvp
from .jax_third_party.svd import svd


def expm(coeff, hmat):
    '''
    Matrix exponential of a Hermitian matrix multiplied by a coefficient.

    Args:
        coeff: float or complex.
        hmat: Hermitian matrix.

    Returns:
        array: matrix exponential of coeff * hmat.
    '''
    e, u = jnp.linalg.eigh(hmat)
    return u.dot(jnp.diag(jnp.exp(coeff * e))).dot(u.transpose().conj())
