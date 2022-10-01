import jax
from jax import lax
from jax import custom_jvp
import jax.numpy as jnp

@custom_jvp
def svd(a):
    return jax.scipy.linalg.svd(a, full_matrices=False, compute_uv=True)

# The code below is copied from in https://github.com/google/jax/blob/main/jax/_src/lax/linalg.py and modified
# In case the above link is dead, checkout 09720b9bcbf07c1318774033f1cb4f4751e37895 and go to the module jax._src.lax.linalg

def _T(x): return jnp.swapaxes(x, -1, -2)
def _H(x): return jnp.conj(_T(x))

@svd.defjvp
def svd_jvp_rule(primals, tangents):
    A, = primals
    dA, = tangents
    U, s, Vt = svd(A)

# MODIFIED SECTION BELOW
    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = jnp.matmul(jnp.matmul(Ut, dA), V)
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
# END OF MODIFIED SECTION

    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
    SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
    dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
    dU = jnp.matmul(U, F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    dV = jnp.matmul(V, F.astype(A.dtype) * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        I = lax.expand_dims(jnp.eye(m, dtype=A.dtype), range(U.ndim - 2))
        dU = dU + jnp.matmul(I - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim.astype(A.dtype)
    if n > m:
        I = lax.expand_dims(jnp.eye(n, dtype=A.dtype), range(V.ndim - 2))
        dV = dV + jnp.matmul(I - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim.astype(A.dtype)

    return (U, s, Vt), (dU, ds, _H(dV))
