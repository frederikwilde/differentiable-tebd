import jax
import jax.numpy as jnp
from jax import custom_jvp


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