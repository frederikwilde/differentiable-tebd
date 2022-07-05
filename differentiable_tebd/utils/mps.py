import numpy as np
import jax
from jax import numpy as jnp
from . import COMPLEX_TYPE
from .linalg import svd


def add_perturbation(mps, perturbation, rng):
    '''
    Add a random perturbation to the mps to avoid degeneracies.

    Args:
        perturbation (float): Factor of random perturbation
            to add.
        rng (np.random.Generator): A RNG created with np.random.default_rng
            Default is None, in which case a fresh PRNG is created.
    Returns:
        array: Perturbed MPS.
    '''
    if rng is None:
        rng = np.random.default_rng()
    r1 = 2 * rng.random(mps.shape) - 1.
    r2 = 2 * rng.random(mps.shape) - 1.
    return mps + perturbation * (r1 + 1.j*r2)

def mps_zero_state(num_sites, chi, perturbation=None, rng=None, d=2):
    '''
    The all zero state: |00000...>

    Args:
        num_sites (int): Number of sites.
        chi (int): Bond dimension.
        perturbation (float): Factor of random perturbation
            to add. Default is None.
        rng (np.random.Generator): A RNG created with np.random.default_rng
            Default is None, in which case a fresh PRNG is created.
        d (int): Physical dimension.
        
    Returns:
        array: MPS
    '''
    mps = jnp.zeros((num_sites, chi, d, chi), dtype=COMPLEX_TYPE)
    for i in range(num_sites):
        mps = mps.at[i, 0, 0, 0].set(1.)
    if perturbation is not None:
        return add_perturbation(mps, perturbation, rng)
    else:
        return mps

def mps_neel_state(num_sites, chi, perturbation=None, rng=None, d=2):
    '''
    Prepare the Neel state as an MPS: |010101...>

    Args:
        num_sites (int): Number of sites.
        chi (int): Bond dimension.
        perturbation (float): Factor of random perturbation
            to add. Default is None.
        rng (np.random.Generator): A RNG created with np.random.default_rng
            Default is None, in which case a fresh PRNG is created.
        d (int): Physical dimension.
        
    Returns:
        array: MPS
    '''
    mps = jnp.zeros((num_sites, chi, d, chi), dtype=COMPLEX_TYPE)
    for i in range(num_sites):
        if i % 2 == 0:
            mps = mps.at[i, 0, 0, 0].set(1.)
        else:
            mps = mps.at[i, 0, 1, 0].set(1.)
    if perturbation is not None:
        return add_perturbation(mps, perturbation, rng)
    else:
        return mps

def contract_and_split(n1, n2, gate):
    chi = n1.shape[2]
    d = n1.shape[1]
    n = jnp.tensordot(n1, n2, axes=(2, 0))
    c = jnp.tensordot(n, gate, axes=((1,2), (2,3))).transpose((0, 2, 3, 1))
    u, s, v = svd(c.reshape(d * chi, d * chi))
    # FIXME: derivative at zero
    s_sqrt = jnp.sqrt(s[:chi])
    truncation_error_squared = jnp.sum(s[chi:] ** 2)
    u_out = (s_sqrt * u[:, :chi]).reshape(chi, d, chi)
    v_out = (s_sqrt * v[:chi, :].transpose()).transpose().reshape(chi, d, chi)
    return u_out, v_out, truncation_error_squared

def apply_gate(mps, i, gate):
    n1, n2, err_sqr = contract_and_split(mps[i], mps[i+1], gate)
    mps = mps.at[i:i+2].set(jnp.array([n1, n2]))
    return mps, err_sqr

def norm_squared(mps):
    '''
    The squared norm of an MPS.
    '''
    def _update_left(left, tensor):
        '''For scan function.'''
        t1 = jnp.tensordot(left, tensor.conj(), axes=(1, 0))
        return jnp.tensordot(tensor, t1, axes=((0,1), (0,1))), None

    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = left.at[0, 0].set(1.)
    left, _ = jax.lax.scan(_update_left, left, mps)
    return left[0,0].real
