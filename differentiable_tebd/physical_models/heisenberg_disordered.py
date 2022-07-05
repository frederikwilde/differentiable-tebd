import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .. import COMPLEX_TYPE
from ..utils import X, Y, Z
from ..utils.linalg import expm
from ..utils.tebd import trotter_step_order2


XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
X1 = np.kron(X, np.eye(2))
X2 = np.kron(np.eye(2), X)
jXX, jYY, jZZ, jX1, jX2, jX = map(jnp.array, [XX, YY, ZZ, X1, X2, X])

# for state vector simulation
def hamiltonian(params, qnum):
    '''
    1-D Heisenberg Hamiltonian with open boundaries (non-periodic) a full
    (exponentially large) matrix.

    Args:
        params (array): Parameters x, y, z, h1, h2, ... hn. Where hi is the local
            field strength in X direction (i.e. 1-body X Paulis) for the i-th qubit.
            x, y, and z determine the strength of the XX, YY, and ZZ interactions,
            respectively.
        qnum (int): Number of sites, or spin-1/2 particles.

    Returns:
        scipy.sparse.csr_matrix: The Hamiltonian.
    '''
    x, y, z = params[:3]
    h = params[3:]
    out = sp.csr_matrix((2**qnum, 2**qnum), dtype=COMPLEX_TYPE)
    for i in range(qnum-1):
        out += x * sp.kron(
            sp.eye(2**i),
            sp.kron(XX, sp.eye(2**(qnum-i-2))))
        out += y * sp.kron(
            sp.eye(2**i),
            sp.kron(YY, sp.eye(2**(qnum-i-2))))
        out += z * sp.kron(
            sp.eye(2**i),
            sp.kron(ZZ, sp.eye(2**(qnum-i-2))))
    for i in range(qnum):
        out += h[i] * sp.kron(
            sp.eye(2**i),
            sp.kron(X, sp.eye(2**(qnum-i-1))))
    return out


def make_gate(h_tuple, x, y, z, deltat):
        h1, h2 = h_tuple
        ham = x*jXX + y*jYY + z*jZZ + .5*(h1*jX1 + h2*jX2)
        return expm(-1.j * deltat, ham).reshape(2,2,2,2)
make_gate_batched = jax.vmap(make_gate, in_axes=(0, None, None, None, None))


@partial(jit, static_argnums=[2])
def mps_evolution_order2(params, deltat, steps, mps):
    '''
    Args:
        params (array): Hamiltonian parameters: x, y, z, h1, ..., hn.
        deltat (float): Trotter step size.
        steps (int): Number of Trotter steps.
        mps (array): Initial mps with shape (num_sites, chi, 2, chi).
    Returns:
        Array: output MPS after steps
        float: Cumulated errors
    '''
    shape = mps.shape
    L, Lh = shape[0], int(shape[0] / 2)
    x, y, z = params[:3]
    h = params[3:]
    odd_layer_gates = make_gate_batched(h[1:L-1].reshape(Lh-1,2), x, y, z, .5*deltat)
    even_layer_gates = make_gate_batched(h[2:L-2].reshape(Lh-2,2), x, y, z, deltat)
    # boundary gates in the even layer
    gate_left = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + h[0]*jX1 + .5*h[1]*jX2).reshape(2,2,2,2)
    gate_right = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + .5*h[-2]*jX1 + h[-1]*jX2).reshape(2,2,2,2)
    
    mps, errors_squared = jax.lax.scan(
            lambda m, _: trotter_step_order2(m, gate_left, even_layer_gates, odd_layer_gates, gate_right),
            mps,
            None,
            length=steps
        )
    return mps, errors_squared
