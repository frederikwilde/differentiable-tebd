import numpy as np
from scipy.integrate import solve_ivp

def vec_simulation(hamiltonian, times, **kwargs):
    '''A crude full-state-vector Schroedinger time evolution solver evolving from
    the |00...0> state.'''
    initial_state = np.zeros(hamiltonian.shape[0], dtype='complex')
    initial_state[0] = 1.
    # time evolution
    sol = solve_ivp(lambda _, v: -1.j*hamiltonian.dot(v), (0., times[-1]), initial_state, t_eval=times, **kwargs)
    if sol['success']:
        print(f"State evolution completed successfully with {sol['nfev']} function calls.")
        return sol['y']
    else:
        msg = sol['message']
        raise RuntimeError(f'State vector evolution was not successful: {msg}')

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
