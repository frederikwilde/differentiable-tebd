'''
Functions provided here are specific to physical dimension d=2.
'''

import jax
import jax.numpy as jnp
from . import Hadamard, YHadamard, COMPLEX_TYPE

def local_basis_transform(mps, basis):
    '''
    Transforms the entire MPS into the X or Y basis. Only for physical dimension d=2.

    Args:
        mps (array): MPS in Z basis.
        basis (int): Either 1 or 2 for transformation into X or Y basis,
            respectively
        
    Returns:
        array: Transformed MPS.
    '''
    return jax.lax.cond(
        basis==1,
        jax.vmap(lambda t: jnp.tensordot(Hadamard, t, axes=(1,1)).transpose((1, 0, 2))),
        jax.vmap(lambda t: jnp.tensordot(YHadamard, t, axes=(1,1)).transpose((1, 0, 2))),
        mps
    )
def _update_left(left, tensors_bit_basis_tuple):
    '''For scan function.'''
    tX, tY, tZ, bit, basis = tensors_bit_basis_tuple
    branches = [
        lambda left: left.dot(tX[:, bit, :]),
        lambda left: left.dot(tY[:, bit, :]),
        lambda left: left.dot(tZ[:, bit, :]),
    ]
    return jax.lax.switch(basis-1, branches, left), None

def probability(mpsX, mpsY, mpsZ, bitstring, pauli_basis, mps_norm_squared=1.):
    '''
    Compute the probability of measuring a given bitstring.

    Args:
        mpsX (array): State in local X basis.
        mpsY (array): State in local Y basis.
        mpsZ (array): State in local Z basis.
        bitstring (array[int]): Measurement outcome.
        pauli_basis (Sequence[int]): Measurement basis. 1 for X, 2 for Y, 3 for Z.
        mps_norm_squared (float): Squared l2 norm of the MPS.
    '''
    left = jnp.zeros(mpsX.shape[1], dtype=COMPLEX_TYPE)
    left = left.at[0].set(1.)
    left, _ = jax.lax.scan(_update_left, left, [mpsX, mpsY, mpsZ, bitstring, pauli_basis])
    return jnp.abs(left[0]) ** 2 / mps_norm_squared
