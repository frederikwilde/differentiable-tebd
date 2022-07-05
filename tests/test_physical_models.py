import unittest
import numpy as np
import jax
import jax.numpy as jnp
import differentiable_tebd.physical_models.heisenberg_disordered as hd
import differentiable_tebd.physical_models.heisenberg as hs
from differentiable_tebd.state_vector_simulation import mps_to_vector, vec_simulation
from differentiable_tebd.utils.mps import mps_zero_state
from differentiable_tebd.utils.mps_qubits import probability, local_basis_transform


def summed_probabilities(mps, bitstrings):
    mpsX = local_basis_transform(mps, 1)
    mpsY = local_basis_transform(mps, 2)
    probs = jax.vmap(probability, in_axes=(None, None, None, 0, 0))(
        mpsX, mpsY, mps, bitstrings, jnp.zeros_like(bitstrings, dtype=int))
    return jnp.sum(probs)


class TestHeisenbergDisordered(unittest.TestCase):
    def setUp(self):
        self.num_sites = 6
        rng = np.random.default_rng(1)
        self.true_params = rng.random(self.num_sites+3) - .5
        self.other_params = jnp.array(rng.random(self.num_sites+3) - .5)
        times = np.array([.2, .4, .6])
        self.states = vec_simulation(hd.hamiltonian(self.true_params, self.num_sites), times, atol=1e-13, rtol=1e-13).T
        self.chi = 10
        self.deltat = .01
        self.steps = 20
        self.bitstrings = jnp.zeros((2, self.num_sites), dtype=int).at[0, 1].set(1)

    def test_heisenberg_disordered(self):
        mps = mps_zero_state(self.num_sites, self.chi)
        mps1, _ = hd.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps)
        mps2, _ = hd.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps1)
        mps3, _ = hd.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps1) - self.states[0]), 0., delta=1e-4)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps2) - self.states[1]), 0., delta=1e-4)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps3) - self.states[2]), 0., delta=1e-4)

    def mock_loss(self, params, perturbation=None):
        mps = mps_zero_state(self.num_sites, self.chi, perturbation)
        mps, _ = hd.mps_evolution_order2(params, self.deltat, self.steps, mps)
        return summed_probabilities(mps, self.bitstrings)

    def test_heisenberg_disordered_grad(self):
        grad = jax.grad(self.mock_loss)(self.other_params, perturbation=1e-6)
        manual_grad = jnp.zeros_like(self.other_params)
        eps = 1e-8
        for i in range(len(manual_grad)):
            f_plus = self.mock_loss(self.other_params.at[i].add(eps))
            f_minus = self.mock_loss(self.other_params.at[i].add(-eps))
            manual_grad = manual_grad.at[i].set((f_plus - f_minus) / (2 * eps))
        assert jnp.allclose(grad, manual_grad, atol=1e-6, rtol=1e-6)


class TestHeisenberg(unittest.TestCase):
    def setUp(self):
        self.num_sites = 6
        rng = np.random.default_rng(1)
        self.true_params = rng.random(4) - .5
        self.other_params = jnp.array(rng.random(4) - .5)
        times = np.array([.2, .4, .6])
        self.states = vec_simulation(hs.hamiltonian(self.true_params, self.num_sites), times, atol=1e-13, rtol=1e-13).T
        self.chi = 10
        self.deltat = .01
        self.steps = 20
        self.bitstrings = jnp.zeros((2, self.num_sites), dtype=int).at[0, 1].set(1)

    def test_heisenberg(self):
        mps = mps_zero_state(self.num_sites, self.chi)
        mps1, _ = hs.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps)
        mps2, _ = hs.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps1)
        mps3, _ = hs.mps_evolution_order2(self.true_params, self.deltat, self.steps, mps2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps1) - self.states[0]), 0., delta=1e-4)    
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps2) - self.states[1]), 0., delta=1e-4)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps3) - self.states[2]), 0., delta=1e-4)
        mps1, _ = hs.mps_evolution(self.true_params, self.deltat, self.steps, mps)
        mps2, _ = hs.mps_evolution(self.true_params, self.deltat, self.steps, mps1)
        mps3, _ = hs.mps_evolution(self.true_params, self.deltat, self.steps, mps2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps1) - self.states[0]), 0., delta=1e-2)    
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps2) - self.states[1]), 0., delta=1e-2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps3) - self.states[2]), 0., delta=1e-2)

    def mock_loss(self, params, perturbation=None):
        mps = mps_zero_state(self.num_sites, self.chi, perturbation)
        mps, _ = hs.mps_evolution_order2(params, self.deltat, self.steps, mps)
        return summed_probabilities(mps, self.bitstrings)

    def test_heisenberg_grad(self):
        grad = jax.grad(self.mock_loss)(self.other_params, perturbation=1e-6)
        manual_grad = jnp.zeros_like(self.other_params)
        eps = 1e-8
        for i in range(len(manual_grad)):
            f_plus = self.mock_loss(self.other_params.at[i].add(eps))
            f_minus = self.mock_loss(self.other_params.at[i].add(-eps))
            manual_grad = manual_grad.at[i].set((f_plus - f_minus) / (2 * eps))
        assert jnp.allclose(grad, manual_grad, atol=1e-6, rtol=1e-6)