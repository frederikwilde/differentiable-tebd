from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
from jax import jit, checkpoint
from jax.ops import index_update, index
from . import COMPLEX_TYPE

rnd = np.random.rand

X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX_TYPE)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)
X_proj_plus = .5 * np.array([[1,1], [1,1]], dtype=COMPLEX_TYPE)
Y_proj_plus = .5 * np.array([[1,-1j], [1j,1]], dtype=COMPLEX_TYPE)
Z_proj_plus = np.array([[1,0], [0,0]], dtype=COMPLEX_TYPE)
X_proj_minus = .5 * np.array([[1,-1], [-1,1]], dtype=COMPLEX_TYPE)
Y_proj_minus = .5 * np.array([[1,1j], [-1j,1]], dtype=COMPLEX_TYPE)
Z_proj_minus = np.array([[0,0], [0,1]], dtype=COMPLEX_TYPE)

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

def mps_zero_state(num_sites, chi, perturbation=None):
    '''
    Args:
        num_sites (int): Number of sites.
        chi (int): Bond dimension.
        perturbation (float): Factor of random perturbation
            to add. Default is None.
    Returns:
        array: MPS
    '''
    mps = jnp.zeros((num_sites, chi, 2, chi), dtype=COMPLEX_TYPE)
    for i in range(num_sites):
        mps = index_update(mps, (i, 0, 0, 0), 1.)
    if perturbation is not None:
        return mps + perturbation * (rnd(*mps.shape) + 1.j*rnd(*mps.shape))
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
    truncation_error = jnp.sum(s[chi:])
    u_out = (s_sqrt * u[:, :chi]).reshape(chi, 2, chi)
    v_out = (s_sqrt * v[:chi, :].transpose()).transpose().reshape(chi, 2, chi)
    return u_out, v_out, truncation_error

def apply_gate(mps, i, gate, truncation_error):
    n1, n2, err = contract_and_split(mps[i], mps[i+1], gate)
    mps = index_update(mps, index[i:i+2], jnp.array([n1, n2]))
    return mps, truncation_error + err

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

branches = [
        lambda x: jnp.tensordot(X_proj_plus, x, axes=(1, 1)),
        lambda x: jnp.tensordot(Y_proj_plus, x, axes=(1, 1)),
        lambda x: jnp.tensordot(Z_proj_plus, x, axes=(1, 1)),
        lambda x: jnp.tensordot(X_proj_minus, x, axes=(1, 1)),
        lambda x: jnp.tensordot(Y_proj_minus, x, axes=(1, 1)),
        lambda x: jnp.tensordot(Z_proj_minus, x, axes=(1, 1))
    ]
def _update_left(left, tensor_bit_basis_tuple):
    t, bit, basis = tensor_bit_basis_tuple
    # index arg of switch is 1, 2, 3 for proj_minus and 4, 5, 6 for proj_plus.
    t1 = jax.lax.switch(3 * bit + basis - 1, branches, t)
    t2 = jnp.tensordot(left, t.conj(), axes=(1, 0))
    return jnp.tensordot(t1, t2, axes=((1,0), (0,1))), None

def probability(mps, bitstring, basis, mps_norm_squared=1.):
    '''
    Compute the probability of measuring a given bitstring.

    Args:
        MPS (array): State which is measured.
        bitstring (array[int]): Measurement outcome.
        basis (Sequence[int]): Measurement basis. 1 for X, 2 for Y, 3 for Z.
        mps_norm_squared (float): Squared l2 norm of the MPS.
    '''
    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = index_update(left, (0, 0), 1.)
    left, _ = jax.lax.scan(_update_left, left, (mps, bitstring, basis))
    return left[0,0].real / mps_norm_squared

### alternative implementation of probability using vmap (is slightly slower, at least on CPU)
# branches2 = [
#         lambda x: jnp.tensordot(X_proj_plus, x, axes=(1, 1)).transpose((1, 0, 2)),
#         lambda x: jnp.tensordot(Y_proj_plus, x, axes=(1, 1)).transpose((1, 0, 2)),
#         lambda x: jnp.tensordot(Z_proj_plus, x, axes=(1, 1)).transpose((1, 0, 2)),
#         lambda x: jnp.tensordot(X_proj_minus, x, axes=(1, 1)).transpose((1, 0, 2)),
#         lambda x: jnp.tensordot(Y_proj_minus, x, axes=(1, 1)).transpose((1, 0, 2)),
#         lambda x: jnp.tensordot(Z_proj_minus, x, axes=(1, 1)).transpose((1, 0, 2))
#     ]
# def _update_left2(left, tensor1_tensor2):
#     t1, t2 = tensor1_tensor2
#     t2 = jnp.tensordot(left, t2.conj(), axes=(1, 0))
#     return jnp.tensordot(t1, t2, axes=((0,1), (0,1))), None

# def _contract_projector(tensor, basis, bit):
#     # index arg of switch is 1, 2, 3 for proj_minus and 4, 5, 6 for proj_plus.
#     return jax.lax.switch(3 * bit + basis - 1, branches2, tensor)

# def probability2(mps, bitstring, basis, mps_norm_squared=1.):
#     '''
#     Compute the probability of measuring a given bitstring.
#     This version copies the entire MPS and contracts the projectors.
#     In principle, this should be faster when the contractions are done
#     in parallel. However, on a CPU the function 'probability' is faster.

#     Args:
#         MPS (array): State which is measured.
#         bitstring (array[int]): Measurement outcome.
#         basis (Sequence[int]): Measurement basis. 1 for X, 2 for Y, 3 for Z.
#         mps_norm_squared (float): Squared l2 norm of the MPS.
#     '''
#     mps_proj = jax.vmap(_contract_projector, in_axes=(0, 0, 0))(mps, basis, bitstring)
#     left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
#     left = index_update(left, (0, 0), 1.)
#     left, _ = jax.lax.scan(_update_left2, left, (mps_proj, mps))
#     return left[0,0].real / mps_norm_squared

### for data only consisting of Z-basis measurements (only marginal advantage)
def _update_left_onlyZ(left, tensor_bit_tuple):
    '''For scan function.'''
    t, b = tensor_bit_tuple
    return left.dot(t[:, b, :]), None

def probability_onlyZ(mps, bitstring):
    '''
    Calculate the probability of measuring a bitstring under the given MPS
    in the Z basis.
    '''
    left = jnp.zeros(mps.shape[1], dtype=COMPLEX_TYPE)
    left = index_update(left, 0, 1.)
    left, _ = jax.lax.scan(_update_left_onlyZ, left, [mps, bitstring])
    return jnp.abs(left[0]) ** 2
