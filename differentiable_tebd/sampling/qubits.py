from functools import lru_cache
import warnings
import scipy
import scipy.sparse as sp
import scipy.stats
import numpy as np
import jax.numpy as jnp
import jax
import pickle
import tensornetwork as tn
from .. import COMPLEX_TYPE

z_proj_0 = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_TYPE)
z_proj_1 = jnp.array([[0, 0], [0, 1]], dtype=COMPLEX_TYPE)
x_proj_0 = jnp.array([[1, 1], [1, 1]], dtype=COMPLEX_TYPE) / 2
x_proj_1 = jnp.array([[1, -1], [-1, 1]], dtype=COMPLEX_TYPE) / 2
y_proj_0 = jnp.array([[1, -1j], [1j, 1]], dtype=COMPLEX_TYPE) / 2
y_proj_1 = jnp.array([[1, 1j], [-1j, 1]], dtype=COMPLEX_TYPE) / 2

dot = jnp.tensordot

def draw_samples_from_mps(mps, basis, key, num_samples):
    '''
    Creates measurement samples in a given basis from the probability-
    distribution defined by an MPS.
    WARNING: If the MPS is not normalized, e.g. because of perturbation
    terms, the samples are not drawn correctly.

    Args:
        mps (jnp.ndarray)
        basis (jnp.ndarray): A list of Pauli bases. Valid entries are 1, 2,
        and 3, for X, Y, and Z respectively.
        key (jnp.ndarray): A JAX PRNG key.
        num_samples (int): Number of strings to sample.

    Returns:
        jnp.ndarray: A new PRNG key.
        jnp.ndarray: Array of samples of shape (num_samples, len(mps))
    '''
    keys = jax.random.split(key, num_samples+1)
    batched_sample = jax.vmap(_one_sample_from_mps, in_axes=(None, None, 0))
    mps = tn.FiniteMPS(mps, canonicalize=True, center_position=0, backend='jax')
    mps = jnp.array(mps.tensors)
    return keys[0], batched_sample(mps, basis, keys[1:])

@jax.jit
def _one_sample_from_mps(mps, basis, key):
    projectors_0 = [
        lambda t: dot(t, x_proj_0, axes=(1, 1)),
        lambda t: dot(t, y_proj_0, axes=(1, 1)),
        lambda t: dot(t, z_proj_0, axes=(1, 1))
    ]
    projectors_1 = [
        lambda t: dot(t, x_proj_1, axes=(1, 1)),
        lambda t: dot(t, y_proj_1, axes=(1, 1)),
        lambda t: dot(t, z_proj_1, axes=(1, 1))
    ]

    def update_left(left, tensor_basis_rnd):
        t, b, r = tensor_basis_rnd
        sandwhich = dot(
            jax.lax.switch(b-1, projectors_0, t),
            t.conj(),
            axes=(2, 1)
        )
        empty_sandwhich = dot(t, t.conj(), axes=(1, 1))
        sandwhich_with_left = dot(left, sandwhich, axes=((0,1), (0,2)))
        probability_0 = jnp.trace(sandwhich_with_left)
        marginal = jnp.trace(dot(left, empty_sandwhich, axes=((0,1), (0,2))))
        def true_fun(_):
            sandwhich = dot(
                jax.lax.switch(b-1, projectors_1, t),
                t.conj(),
                axes=(2, 1)
            )
            return dot(left, sandwhich, axes=((0,1), (0,2))), 1
        return jax.lax.cond(
            r > probability_0 / marginal,
            true_fun,
            lambda _: (sandwhich_with_left, 0),
            ()
        )
        
    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = left.at[0, 0].set(1.)
    rnd = jax.random.uniform(key, [mps.shape[0]])
    _, bitstring = jax.lax.scan(update_left, left, [mps, basis, rnd])
    return bitstring

def save_basis_transforms(dir, num_sites):
    '''Pickles all basis transforms to specified directory.'''
    for i in range(num_sites):
        left_half = sp.kron(sp.eye(2**i), 1/np.sqrt(2) * np.array([[1,1], [1,-1]], dtype=COMPLEX_TYPE))
        t = sp.kron(left_half, sp.eye(2**(num_sites-i-1))).asformat('csr')
        with open(f'{dir}/x{i}.pickle', 'xb') as f:
            pickle.dump(t, f, protocol=4)
        left_half = sp.kron(sp.eye(2**i), .5 * np.array([[1+1j,1-1j], [1-1j, 1+1j]], dtype=COMPLEX_TYPE))
        t = sp.kron(left_half, sp.eye(2**(num_sites-i-1))).asformat('csr')
        with open(f'{dir}/y{i}.pickle', 'xb') as f:
            pickle.dump(t, f, protocol=4)

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