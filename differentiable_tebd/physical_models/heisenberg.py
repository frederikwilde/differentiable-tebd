import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import jit, checkpoint
from jax.ops import index_update, index
from functools import partial
from .. import COMPLEX_TYPE
from ..mps_utils import (
    X, Y, Z,
    apply_gate,
    contract_and_split,
    expm,
)

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

@checkpoint
def trotter_step(mps, gate_left, gate_middle, gate_right):
    '''
    Number of qubits must be even!
    Applies one Trotter step to mps.
    gate1 is applied to all but the last pair of qubits.
    gate2 is only applied to the last pair as displayed below.
    |  |  |  |  |  |  |  |
    [gl]  [gm]  [gm]  [gr]
    |  [gm]  [gm]  [gm]  |
    |  |  |  |  |  |  |  |

    Returns:
        array: New MPS.
        float: Truncation error.
    '''
    shape = mps.shape
    L, Lh = shape[0], int(shape[0] / 2)
    trunc_err_sqr = 0.

    def apply_gate_totuple(tensor_tuple):
        t1, t2 = tensor_tuple
        n1, n2, err_sqr = contract_and_split(t1, t2, gate_middle)
        return jnp.stack([n1, n2]), err_sqr
    batched_apply_gate = jax.vmap(apply_gate_totuple)

    # even layer
    mps, err_sqr = apply_gate(mps, 0, gate_left)
    trunc_err_sqr += err_sqr
    middle_tensors, errs_sqr = batched_apply_gate(
        mps[2:L-2].reshape(Lh-2, 2, *shape[1:])
    )
    mps = index_update(mps, index[2:L-2], middle_tensors.reshape(L-4, *shape[1:]))
    trunc_err_sqr += jnp.sum(errs_sqr)
    mps, err_sqr = apply_gate(mps, L-2, gate_right)
    trunc_err_sqr += err_sqr

    # odd layer
    middle_tensors, errs_sqr = batched_apply_gate(
        mps[1:L-1].reshape(Lh-1, 2, *shape[1:])
    )
    mps = index_update(mps, index[1:L-1], middle_tensors.reshape(L-2, *shape[1:]))
    trunc_err_sqr += jnp.sum(errs_sqr)

    return mps, trunc_err_sqr

@checkpoint
def trotter_step_order2(mps, gate_left, gate_middle, gate_middle_halftime, gate_right):
    '''
    Number of qubits must be even!
    Applies one 2nd order Trotter step to mps.

    |  |   |  |   |  |   |  |
    |  [gmh]  [gmh]  [gmh]  |
    [gl]   [gm]   [gm]   [gr]
    |  [gmh]  [gmh]  [gmh]  |
    |  |   |  |   |  |   |  |

    Args:
        mps: Input MPS.
        gate_left: is applied to the first and second qubit.
            Involves one time step deltat.
        gate_right: is applied to the second to last and last qubit.
            Involves one tim estep deltat.
        gate_middle: is applied to all other qubits.
            Involves one time step deltat.
        gate_middle_halftime: the same as gate_middle but only
            involves half a time step deltat/2.
    '''
    shape = mps.shape
    L, Lh = shape[0], int(shape[0] / 2)
    trunc_err_sqr = 0.

    def apply_gate_totuple(tensor_tuple, gate):
        t1, t2 = tensor_tuple
        n1, n2, err_sqr = contract_and_split(t1, t2, gate)
        return jnp.stack([n1, n2]), err_sqr
    batched_apply_gate = jax.vmap(lambda t: apply_gate_totuple(t, gate_middle))
    batched_apply_gate_halftime = jax.vmap(lambda t: apply_gate_totuple(t, gate_middle_halftime))

    # odd layer
    middle_tensors, errs_sqr = batched_apply_gate_halftime(
        mps[1:L-1].reshape(Lh-1, 2, *shape[1:])
    )
    mps = index_update(mps, index[1:L-1], middle_tensors.reshape(L-2, *shape[1:]))
    trunc_err_sqr += jnp.sum(errs_sqr)
    
    # even layer
    mps, err_sqr = apply_gate(mps, 0, gate_left)
    trunc_err_sqr += err_sqr
    middle_tensors, errs_sqr = batched_apply_gate(
        mps[2:L-2].reshape(Lh-2, 2, *shape[1:])
    )
    mps = index_update(mps, index[2:L-2], middle_tensors.reshape(L-4, *shape[1:]))
    trunc_err_sqr += jnp.sum(errs_sqr)
    mps, err_sqr = apply_gate(mps, L-2, gate_right)
    trunc_err_sqr += err_sqr

    # odd layer
    middle_tensors, errs_sqr = batched_apply_gate_halftime(
        mps[1:L-1].reshape(Lh-1, 2, *shape[1:])
    )
    mps = index_update(mps, index[1:L-1], middle_tensors.reshape(L-2, *shape[1:]))
    trunc_err_sqr += jnp.sum(errs_sqr)

    return mps, trunc_err_sqr

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
