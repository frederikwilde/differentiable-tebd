import numpy as np
import jax
from jax import custom_jvp
from jax import numpy as jnp
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

def mps_zero_state(num_sites, chi, perturbation=None, rng=None):
    '''
    The all zero state: |00000...>

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
        mps = mps.at[i, 0, 0, 0].set(1.)
    if perturbation is not None:
        return add_perturbation(mps, perturbation, rng)
    else:
        return mps

def mps_neel_state(num_sites, chi, perturbation=None, rng=None):
    '''
    Prepare the Neel state as an MPS: |010101...>

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
        if i % 2 == 0:
            mps = mps.at[i, 0, 0, 0].set(1.)
        else:
            mps = mps.at[i, 0, 1, 0].set(1.)
    if perturbation is not None:
        return add_perturbation(mps, perturbation, rng)
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

@custom_jvp
def svd(a):
    return jax.scipy.linalg.svd(a, full_matrices=False, compute_uv=True)

def _T(x): return jnp.swapaxes(x, -1, -2)
def _H(x): return jnp.conj(_T(x))

@svd.defjvp
def svd_jvp_rule(primals, tangents):
    A, = primals
    dA, = tangents
    U, s, Vt = svd(A)

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = jnp.matmul(jnp.matmul(Ut, dA), V)
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
    # s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim * dS  # dS.dot(jnp.diag(s))
    SdS = _T(s_dim) * dS  # jnp.diag(s).dot(dS)

    s_zeros = jnp.ones((), dtype=A.dtype) * (s == 0.)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
    dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat
    dU = jnp.matmul(U, F * (dSS + _H(dSS)) + dUdV_diag)
    dV = jnp.matmul(V, F * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        I = jax.lax.expand_dims(jnp.eye(m, dtype=A.dtype), range(U.ndim - 2))
        dU = dU + jnp.matmul(I - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim
    if n > m:
        I = jax.lax.expand_dims(jnp.eye(n, dtype=A.dtype), range(V.ndim - 2))
        dV = dV + jnp.matmul(I - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim

    return (U, s, Vt), (dU, ds, _H(dV))


def contract_and_split(n1, n2, gate):
    chi = n1.shape[2]
    n = jnp.tensordot(n1, n2, axes=(2, 0))
    c = jnp.tensordot(n, gate, axes=((1,2), (2,3))).transpose((0, 2, 3, 1))
    u, s, v = svd(c.reshape(2*chi, 2*chi))
    # FIXME: derivative at zero
    s_sqrt = jnp.sqrt(s[:chi])
    truncation_error_squared = jnp.sum(s[chi:] ** 2)
    u_out = (s_sqrt * u[:, :chi]).reshape(chi, 2, chi)
    v_out = (s_sqrt * v[:chi, :].transpose()).transpose().reshape(chi, 2, chi)
    return u_out, v_out, truncation_error_squared

def apply_gate(mps, i, gate):
    n1, n2, err_sqr = contract_and_split(mps[i], mps[i+1], gate)
    mps = mps.at[i:i+2].set(jnp.array([n1, n2]))
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
    left = left.at[0, 0].set(1.)
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
    left = left.at[0].set(1.)
    left, _ = jax.lax.scan(_update_left, left, [mpsX, mpsY, mpsZ, bitstring, basis])
    return jnp.abs(left[0]) ** 2 / mps_norm_squared
