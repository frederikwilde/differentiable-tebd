import numpy as np
import jax
from jax import numpy as jnp
from jax.ops import index_update, index
from . import COMPLEX_TYPE

X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX_TYPE)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)
Hadamard = np.array([[1,1], [1, -1]], dtype=COMPLEX_TYPE) / np.sqrt(2.)
YHadamard = np.array([[1,-1j], [1,1j]], dtype=COMPLEX_TYPE) / np.sqrt(2.)


class HashableArray(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        return obj
    def __hash__(self):
        return self.__str__().__hash__()
    def __eq__(self, other):
        return jnp.all(jnp.array(self) == jnp.array(other))
    def __ne__(self, other):
        return jnp.any(jnp.array(self) != jnp.array(other))

def mps_zero_state(num_sites, chi, perturbation=None, rng=None):
    '''
    Args:
        num_sites (int): Number of sites.
        chi (int): Bond dimension.
        perturbation (float): Factor of random perturbation
            to add. Default is None.
        rng (np.random.Generator): A RNG created with np.random.default_rng
            Default is None, in which case a fresh PRNG is created.
        
    Returns:
        array: MPS
    '''
    mps = jnp.zeros((num_sites, chi, 2, chi), dtype=COMPLEX_TYPE)
    for i in range(num_sites):
        mps = index_update(mps, (i, 0, 0, 0), 1.)
    if perturbation is not None:
        if rng is None:
            rng = np.random.default_rng()
        r1 = 2 * rng.random(mps.shape) - 1.
        r2 = 2 * rng.random(mps.shape) - 1.
        return mps + perturbation * (r1 + 1.j*r2)
    else:
        return mps

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

def contract_and_split(n1, n2, gate):
    chi = n1.shape[2]
    n = jnp.tensordot(n1, n2, axes=(2, 0))
    c = jnp.tensordot(n, gate, axes=((1,2), (2,3))).transpose((0, 2, 3, 1))
    u, s, v =  jax.scipy.linalg.svd(c.reshape(2*chi, 2*chi), full_matrices=False)
    # FIXME: derivative at zero
    s_sqrt = jnp.sqrt(s[:chi])
    truncation_error_squared = jnp.sum(s[chi:] ** 2)
    u_out = (s_sqrt * u[:, :chi]).reshape(chi, 2, chi)
    v_out = (s_sqrt * v[:chi, :].transpose()).transpose().reshape(chi, 2, chi)
    return u_out, v_out, truncation_error_squared

def apply_gate(mps, i, gate):
    n1, n2, err_sqr = contract_and_split(mps[i], mps[i+1], gate)
    mps = index_update(mps, index[i:i+2], jnp.array([n1, n2]))
    return mps, err_sqr

### MPS contractions
def norm_squared(mps):
    '''
    The squared norm of an MPS.
    '''
    def _update_left(left, tensor):
        '''For scan function.'''
        t1 = jnp.tensordot(left, tensor.conj(), axes=(1, 0))
        return jnp.tensordot(tensor, t1, axes=((0,1), (0,1))), None

    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = index_update(left, (0, 0), 1.)
    left, _ = jax.lax.scan(_update_left, left, mps)
    return left[0,0].real

def local_basis_transform(mps, basis):
    '''
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

def probability(mpsX, mpsY, mpsZ, bitstring, basis, mps_norm_squared=1.):
    '''
    Compute the probability of measuring a given bitstring.

    Args:
        mpsX (array): State in local X basis.
        mpsY (array): State in local Y basis.
        mpsZ (array): State in local Z basis.
        bitstring (array[int]): Measurement outcome.
        basis (Sequence[int]): Measurement basis. 1 for X, 2 for Y, 3 for Z.
        mps_norm_squared (float): Squared l2 norm of the MPS.
    '''
    left = jnp.zeros(mpsX.shape[1], dtype=COMPLEX_TYPE)
    left = index_update(left, 0, 1.)
    left, _ = jax.lax.scan(_update_left, left, [mpsX, mpsY, mpsZ, bitstring, basis])
    return jnp.abs(left[0]) ** 2 / mps_norm_squared
