import jax
import jax.numpy as jnp
from jax import checkpoint
from .mps import contract_and_split, apply_gate


def apply_gate_totuple(tensor_tuple, gate):
    t1, t2 = tensor_tuple
    n1, n2, err_sqr = contract_and_split(t1, t2, gate)
    return jnp.stack([n1, n2]), err_sqr


def apply_gate_batched(mps, i_from, i_to, gate):
    '''
    Applies gate or gates to the MPS within a specified index range.
    
    Args:
        mps (array): MPS
        i_from (int): Index of the first MPS tensor to apply the gate to.
        i_to (int): Index of the first MPS tensor not to apply the gate to.
        gate (array): Either a 4-dimensional array, which is applied to all
            specified MPS tensors, or a 5-dimensional array with the first
            dimension being equal to the half the number of MPS tensors specified
            by [i_from: i_to].
    
    Returns:
        array: The MPS after contractions.
        float: The summed squares of the truncation errors.
    '''
    if mps.shape[0] % 2 != 0:
        raise ValueError('The number of sites must be even.')
    K = (i_to - i_from)
    if K % 2 != 0:
        raise ValueError('[i_from: i_to] must describe a slice of even length.')

    if len(gate.shape) == 4:
        apply = jax.vmap(apply_gate_totuple, in_axes=(0, None))
    elif len(gate.shape) == 5:
        apply = jax.vmap(apply_gate_totuple, in_axes=(0, 0))
    else:
        raise ValueError(f'gate has invalid shape. Should be either 4, or 5 dimensional, was {gate.shape}.')

    tensors, errs_sqr = apply(
        mps[i_from:i_to].reshape(K//2, 2, *mps.shape[1:]),
        gate
    )
    mps = mps.at[i_from:i_to].set(tensors.reshape(K, *mps.shape[1:]))
    return mps, jnp.sum(errs_sqr)


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

    Args:
        mps (array): Input MPS.
        gate_left (array): is applied to the first and second qubit.
            Involves one time step deltat.
        gate_middle (array): is applied to all other qubits.
            Involves one time step deltat. Can be a list of gates.
        gate_right (array): is applied to the second to last and last qubit.
            Involves one time step deltat.

    Returns:
        array: New MPS.
        float: Summed squared truncation errors.
    '''
    L = mps.shape[0]
    if L % 2 != 0:
        raise ValueError('The number of sites must be even.')
    trunc_err_sqr = 0.

    # even layer
    mps, err_sqr = apply_gate(mps, 0, gate_left)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate_batched(mps, 2, L-2, gate_middle)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate(mps, L-2, gate_right)
    trunc_err_sqr += err_sqr

    # odd layer
    mps, err_sqr = apply_gate_batched(mps, 1, L-1, gate_middle)
    trunc_err_sqr += err_sqr

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
        mps (array): Input MPS.
        gate_left (array): is applied to the first and second qubit.
            Involves one time step deltat.
        gate_middle (array): is applied to all other qubits.
            Involves one time step deltat. Can be a list of gates.
        gate_middle_halftime (array): the same as gate_middle but only
            involves half a time step deltat/2. Can be a list of gates.
        gate_right (array): is applied to the second to last and last qubit.
            Involves one time step deltat.
    
    Returns:
        array: Output MPS.
        float: Summed squared errors.
    '''
    L = mps.shape[0]
    if L % 2 != 0:
        raise ValueError('The number of sites must be even.')
    trunc_err_sqr = 0.

    # odd layer
    mps, err_sqr = apply_gate_batched(mps, 1, L-1, gate_middle_halftime)
    trunc_err_sqr += err_sqr

    # even layer
    mps, err_sqr = apply_gate(mps, 0, gate_left)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate_batched(mps, 2, L-2, gate_middle)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate(mps, L-2, gate_right)
    trunc_err_sqr += err_sqr

    # odd layer
    mps, err_sqr = apply_gate_batched(mps, 1, L-1, gate_middle_halftime)
    trunc_err_sqr += err_sqr

    return mps, trunc_err_sqr
