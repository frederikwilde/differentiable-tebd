import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .. import COMPLEX_TYPE
from ..utils import X, Y, Z
from ..utils.linalg import expm
from ..utils.tebd import trotter_step, trotter_step_order2


XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
X1 = np.kron(X, np.eye(2))
X2 = np.kron(np.eye(2), X)
jXX, jYY, jZZ, jX1, jX2, jX = map(jnp.array, [XX, YY, ZZ, X1, X2, X])


def hamiltonian(params, qnum):
    '''
    1-D Heisenberg Hamiltonian with open boundaries (non-periodic) a full
    (exponentially large) matrix.

    Args:
        params (array): Parameters h, x, y, z. Where h is the field strength
            in X direction (i.e. 1-body X Paulis). x, y, and z determine the
            strength of the XX, YY, and ZZ interactions, respectively.
        qnum (int): Number of sites, or spins.

    Returns:
        scipy.sparse.csr_matrix: The Hamiltonian.
    '''
    h, x, y, z = params
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
        out += h * sp.kron(
            sp.eye(2**i),
            sp.kron(X, sp.eye(2**(qnum-i-1))))
    return out

@partial(jit, static_argnums=[2])
def mps_evolution(params, deltat, steps, mps):
    '''
    Args:
        params (array): Hamiltonian parameters.
        deltat (float): Trotter step size.
        steps (int): Number of Trotter steps.
        mps (array): Initial mps with shape (num_sites, chi, 2, chi).
    Returns:
        Array: output MPS after steps
        float: Cumulated errors
    '''
    h, x, y, z = params
    gate_left = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + h*(jX1 + .5*jX2)).reshape(2,2,2,2)
    gate_middle = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + .5*h*(jX1 + jX2)).reshape(2,2,2,2)
    gate_right = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + h*(.5*jX1 + jX2)).reshape(2,2,2,2)
    
    mps, errors_squared = jax.lax.scan(
            lambda m, _: trotter_step(m, gate_left, gate_middle, gate_right),
            mps,
            None,
            length=steps
        ) 
    return mps, errors_squared

@partial(jit, static_argnums=[2])
def mps_evolution_order2(params, deltat, steps, mps):
    '''
    Args:
        params (array): Hamiltonian parameters.
        deltat (float): Trotter step size.
        steps (int): Number of Trotter steps.
        mps (array): Initial mps with shape (num_sites, chi, 2, chi).
    Returns:
        Array: output MPS after steps
        float: Cumulated errors
    '''
    h, x, y, z = params
    gate_left = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + h*(jX1 + .5*jX2)).reshape(2,2,2,2)
    gate_middle = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + .5*h*(jX1 + jX2)).reshape(2,2,2,2)
    gate_middle_halftime = expm(-.5j * deltat,
        x*jXX + y*jYY + z*jZZ + .5*h*(jX1 + jX2)).reshape(2,2,2,2)
    gate_right = expm(-1.j * deltat,
        x*jXX + y*jYY + z*jZZ + h*(.5*jX1 + jX2)).reshape(2,2,2,2)
    
    mps, errors_squared = jax.lax.scan(
            lambda m, _: trotter_step_order2(m, gate_left, gate_middle, gate_middle_halftime, gate_right),
            mps,
            None,
            length=steps
        )
    return mps, errors_squared
