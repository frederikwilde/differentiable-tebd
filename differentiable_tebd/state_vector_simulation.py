import numpy as np
from scipy.integrate import solve_ivp

def vec_simulation(hamiltonian, times, initial_state=None, **kwargs):
    '''
    A crude full-state-vector Schroedinger time evolution solver.

    Args:
        hamiltonian (np.ndarray): Hermitian n by n array.
        times (list[float]): A list of times at which the state vector
            is to be recorded. The last value determines the length of
            the time evolution.
        initial_state (np.ndarray or None): Default is None. In this case
            the all zero state |00...0> is the initial state vector.
            Otherwise a normalized array of length n must be passed.
        kwargs: Directly passed to scipy.integrate.solve_ivp

    Returns:
        np.ndarray: A transposed list of state vectors at the specified times.
    '''
    if initial_state is None:
        initial_state = np.zeros(hamiltonian.shape[0], dtype=hamiltonian.dtype)
        initial_state[0] = 1.
    sol = solve_ivp(lambda _, v: -1.j*hamiltonian.dot(v), (0., times[-1]), initial_state, t_eval=times, **kwargs)
    if sol['success']:
        print(f"State evolution completed successfully with {sol['nfev']} function calls.")
        return sol['y']
    else:
        raise RuntimeError(f"State vector evolution was not successful: {sol['message']}")

def mps_to_vector(mps):
    '''Convert MPS to full state vector.'''
    num_sites = mps.shape[0]
    bitstrings = np.zeros((2**num_sites, num_sites), dtype=int)
    for i in range(num_sites):
        bitstrings[:, i] = np.tile(np.repeat(np.array([0, 1]), 2**(num_sites-i-1)), 2**i)
    vector = np.ndarray(2**num_sites, dtype='complex')
    for i in range(2**num_sites):
        contraction = mps[0, :, bitstrings[i, 0], :]
        for j in range(1, num_sites):
            contraction = contraction.dot(mps[j, :, bitstrings[i, j], :])
        vector[i] = contraction[0, 0]
    return vector
