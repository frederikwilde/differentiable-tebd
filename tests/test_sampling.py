import unittest
import numpy as np
import jax
import jax.numpy as jnp
import os
import shutil
from differentiable_tebd import sampling
from differentiable_tebd.mps_utils import mps_zero_state
from differentiable_tebd.state_vector_simulation import mps_to_vector
from differentiable_tebd.physical_models.heisenberg_disordered import mps_evolution_order2


def rng():
    '''Convenience function for creating a fresh seed-12 rng.'''
    return np.random.default_rng(12)

class TestDrawSamples(unittest.TestCase):
    def setUp(self):
        vec1 = np.array([0, 0, 1, 0], dtype=np.complex128)
        basis1 = np.array([3, 3])
        result1 = np.array([1, 0])
        vec2 = np.full(8, 1j/np.sqrt(8), dtype=np.complex128)
        basis2 = np.array([1, 1, 1])
        result2 = np.array([0, 0, 0])
        rng = np.random.default_rng(42)
        # intentionally not normalized
        self.vec3 = rng.random(2**5, dtype=np.float64) + \
            1j * rng.random(2**5, dtype=np.float64)
        self.basis3 = rng.integers(1, 4, 5)
        self.result3_seed12 = np.array([[0, 0, 1, 1, 0],
                                        [1, 1, 1, 1, 0],
                                        [0, 0, 1, 1, 0],
                                        [0, 0, 1, 0, 1],
                                        [0, 1, 0, 1, 0],
                                        [0, 0, 1, 1, 0],
                                        [1, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [1, 1, 0, 1, 0],
                                        [1, 1, 0, 1, 0]])
        self.vecs = [vec1, vec2, self.vec3]
        self.bases = [basis1, basis2, self.basis3]
        self.results = [result1, result2, self.result3_seed12[0]]

    def test_draw_samples(self):
        for v, b, r in zip(self.vecs, self.bases, self.results):
            out = sampling.draw_samples(v, b, 1, rng=rng())
            np.testing.assert_array_equal(out[0], r)
    
    def test_batch(self):
        out = sampling.draw_samples(self.vec3, self.basis3, 10, rng=rng())
        np.testing.assert_array_equal(out, self.result3_seed12)
    
    def test_sequential(self):
        out = sampling.draw_samples(self.vec3, self.basis3, 10, rng=rng(), sequential_samples=True)
        np.testing.assert_array_equal(out, self.result3_seed12)
    
    def test_pickled_transformations(self):
        trafo_dir = 'test_basis_transformations_84758903874982902843993'
        os.mkdir(trafo_dir)
        sampling.save_basis_transforms(trafo_dir, 5)
        out = sampling.draw_samples(self.vec3, self.basis3, 10, rng=rng(), basis_transforms_dir=trafo_dir)
        np.testing.assert_array_equal(out, self.result3_seed12)
        shutil.rmtree(trafo_dir)

class TestDrawSamplesMps(unittest.TestCase):
    def test_draw_samples(self):
        self.num_sites = 4
        basis = jnp.array([1,2,3,1])
        mps = mps_zero_state(self.num_sites, 10)
        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (self.num_sites+3,))
        mps, _ = mps_evolution_order2(params, .1, 10, mps)
        vec = mps_to_vector(mps)
        prob_distribution = sampling.draw_samples(vec, basis.tolist(), 0)
        _, samples = sampling.draw_samples_from_mps(mps, basis, key, 1000)
        hist = self.samples_to_histogram(samples)
        self.assertLess(self.kl_divergence(prob_distribution, hist), 1e-2)

    @staticmethod
    def kl_divergence(d1: np.ndarray, d2: np.ndarray) -> float:
        return np.sum(d1 * np.log(d1 / d2))

    def samples_to_histogram(self, samples):
        non_binary = np.sum(np.array(samples) * 2 ** np.arange(self.num_sites-1, -1, -1), axis=1)
        bins = np.arange(-.5, 2**self.num_sites+.5, 1.)
        return np.histogram(non_binary, bins=bins, density=True)[0]

class TestUnique(unittest.TestCase):
    pass
