from functools import lru_cache
import warnings
import scipy
import scipy.sparse as sp
import scipy.stats
import numpy as np
import jax.numpy as jnp
import pickle
from . import COMPLEX_TYPE

def samples_from_mps(mps):
    '''
    Contracts the MPS with the specified projector and the Hermition conjugate MPS.

    Args:
        mps (array): MPS with dimensions (num_sites, chi, physical_dim, chi)
    '''
    raise NotImplementedError('Sampling from MPS is not implemented yet. See DOI: 10.1103/PhysRevX.8.031012 for details.')
    # def _update_left(left, tensor_contractbool_tuple):
    #     '''For scan function.'''
    #     t, contract = tensor_contractbool_tuple
    #     t1 = jax.lax.cond(
    #         contract,
    #         lambda x: jnp.tensordot(x, projector, axes=(1, 0)).transpose((0, 2, 1)),
    #         lambda x: x,
    #         t)
    #     tt = jnp.tensordot(t1, t.conj(), axes=(1, 1))
    #     return jnp.tensordot(left, tt, axes=((0,1), (0,2))), None

    # L = mps.shape[0]
    # probs = jnp.zeros(L, dtype=jnp.float64)
    # contractbools = jnp.full(L, False)
    # for i in range(L):
    #     left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    #     left = index_update(left, [0, 0], 1.)
    #     left, _ = jax.lax.scan(_update_left, left, [mps, index_update(contractbools, i, True)])
    #     probs = index_update(probs, i, left[0,0].real / mps_norm_squared)  
    #     print(left[0,0].imag)
    # return probs

def save_basis_transforms(dir, num_sites):
    '''Pickles all basis transforms to specified directory.'''
    for i in range(num_sites):
        left_half = sp.kron(sp.eye(2**i), 1/np.sqrt(2) * np.array([[1,1], [1,-1]], dtype=COMPLEX_TYPE))
        t = sp.kron(left_half, sp.eye(2**(num_sites-i-1))).asformat('csr')
        with open(f'{dir}/x{i}.pickle', 'xb') as f:
            pickle.dump(t, f)
        left_half = sp.kron(sp.eye(2**i), .5 * np.array([[1+1j,1-1j], [1-1j, 1+1j]], dtype=COMPLEX_TYPE))
        t = sp.kron(left_half, sp.eye(2**(num_sites-i-1))).asformat('csr')
        with open(f'{dir}/y{i}.pickle', 'xb') as f:
            pickle.dump(t, f)

@lru_cache(1000)
def basis_transform_generic(basis, site, num_sites):
    '''Returns a local basis transformation (as a sparse matrix) on
    site i in a (2 ** num_sites)-dimensional system.'''
    if site >= num_sites:
        raise ValueError(f'Position was {site}, but must be smaller than num_sites {num_sites}')
    if basis == 1:
        left_half = sp.kron(sp.eye(2**site), 1/np.sqrt(2) * np.array([[1,1], [1,-1]], dtype=COMPLEX_TYPE))
        return sp.kron(left_half, sp.eye(2**(num_sites-site-1))).asformat('csr')
    elif basis == 2:
        left_half = sp.kron(sp.eye(2**site), .5 * np.array([[1+1j,1-1j], [1-1j, 1+1j]], dtype=COMPLEX_TYPE))
        return sp.kron(left_half, sp.eye(2**(num_sites-site-1))).asformat('csr')
    else:
        raise ValueError(f'basis must be x or y, was {basis}.')

def draw_samples(vec, basis, num_samples, sequential_samples=False, **kwargs):
    '''
    Generate a number of measurement results, as bit strings, given a state vector.

    Args:
        vec (array): state vector which is sampled.
        basis (array): A basis for each site to measure in. Valid entries are 1, 2,
            and 3 for X, Y, and Z measurements, respectively.
        num_samples (int): Number of samples to draw. If non-positive integer is given
            the full distribution in the given basis is returned.
        sequential_samples (bool): If true the samples are drawn sequentially. This
            can be necessary if the vector or the number of samples is so large, that
            not enough memory is available to allocate a (num_samples, len(vec)) shape
            array.
    
    Kwargs:
        rng (np.random.Generator): A RNG created with np.random.default_rng(seed).
            Note that seed is an optional argument. If not provided, a fresh PRNG is
            created.
        basis_transforms_dir (str): A directory containing pickled basis transformations
            as 'x0.pickle', 'x1.pickle', ... and 'y0.pickle' and so on.
    
    Returns:
        array: A list of bitstrings.
    '''
    num_sites = int(np.log2(vec.size))
    # define basis transform function
    if 'basis_transforms_dir' in kwargs:
        d = kwargs['basis_transforms_dir']
        def basis_transform(basis, site):
            '''Returns a local basis transformation (as a sparse matrix) on
            site i in a (2 ** num_sites)-dimensional system.'''
            if site >= num_sites:
                raise ValueError(f'Position was {site}, but must be smaller than num_sites {num_sites}')
            if basis == 1:
                with open(f'{d}/x{site}.pickle', 'rb') as f:
                    return pickle.load(f)
            elif basis == 2:
                with open(f'{d}/y{site}.pickle', 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f'basis must be x or y, was {basis}.')
    else:
        basis_transform = lambda basis, site: basis_transform_generic(basis, site, num_sites)
    # draw samples
    for i in range(num_sites):
        if basis[i] == 3:
            pass
        else:
            vec = basis_transform(basis[i], i).dot(vec)
    rng = kwargs.get('rng', np.random.default_rng())
    if hasattr(kwargs, 'seed'):
        warnings.warn('''The seed keyword argument is ignored. Instead supply a np.random.Generator
under the keyword argument rng.''', warnings.DeprecationWarning)
    distribution = np.abs(vec) ** 2
    distribution = distribution / distribution.sum()
    if num_samples < 1:
        return distribution
    else:
        randvar = scipy.stats.rv_discrete(values=(np.arange(2**num_sites), distribution))
        out = np.zeros((num_samples, num_sites), dtype=int)
        if sequential_samples:
            for i in range(num_samples):
                s = randvar.rvs(size=1, random_state=rng)[0]
                for j, b in enumerate(np.binary_repr(s)[::-1]):
                    out[i, -1*(j+1)] = int(b)
        else:
            samples = randvar.rvs(size=num_samples, random_state=rng)
            for i, s in enumerate(samples):
                for j, b in enumerate(np.binary_repr(s)[::-1]):
                    out[i, -1*(j+1)] = int(b)
        return out
        # bitstrings = np.zeros((2**num_sites, num_sites), dtype=int)
        # for i in range(num_sites):
        #     bitstrings[:, i] = np.tile(np.repeat(np.array([0, 1]), 2**(num_sites-i-1)), 2**i)
        # return bitstrings[samples]

def unique(strings):
    '''
    Finds all unique strings (given as a sequence of numbers)
    in a given list of strings.

    Args:
        strings (array): Should be an m by n matrix
            where m is the number of strings and n is the length
            of each string.
    
    Returns:
        histogram (array[int]): The number each unique string
            occurs in the input.
        unique_strings (array): The list of unique strings.
    '''
    unique_strings = np.unique(strings, axis=0)
    histogram = []
    shp = strings.shape
    for us in unique_strings:
        histogram.append(
            np.all(strings == np.tile(us, shp[0]).reshape(*shp), axis=1).sum()
        )
    return jnp.array(histogram), jnp.array(unique_strings)