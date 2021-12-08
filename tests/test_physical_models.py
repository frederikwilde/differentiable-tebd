import unittest
import numpy as np
import differentiable_tebd.physical_models.heisenberg_disordered as hd
import differentiable_tebd.physical_models.heisenberg as hs
from differentiable_tebd.state_vector_simulation import mps_to_vector, vec_simulation
from differentiable_tebd.mps_utils import mps_zero_state

class TestHeisenbergDisordered(unittest.TestCase):
    def test_heisenberg_disordered(self):
        num_sites = 6
        rng = np.random.default_rng(1)
        true_params = rng.random(num_sites+3) - .5
        times = np.array([.2, .4, .6])
        states = vec_simulation(hd.hamiltonian(true_params, num_sites), times, atol=1e-13, rtol=1e-13).T
        chi = 10
        deltat = .01
        steps = 20
        mps = mps_zero_state(num_sites, chi)
        mps1, _ = hd.mps_evolution_order2(true_params, deltat, steps, mps)
        mps2, _ = hd.mps_evolution_order2(true_params, deltat, steps, mps1)
        mps3, _ = hd.mps_evolution_order2(true_params, deltat, steps, mps2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps1) - states[0]), 0., delta=1e-4)    
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps2) - states[1]), 0., delta=1e-4)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps3) - states[2]), 0., delta=1e-4)

    def test_heisenberg(self):
        num_sites = 6
        rng = np.random.default_rng(1)
        true_params = rng.random(4) - .5
        times = np.array([.2, .4, .6])
        states = vec_simulation(hs.hamiltonian(true_params, num_sites), times, atol=1e-13, rtol=1e-13).T
        chi = 10
        deltat = .01
        steps = 20
        mps = mps_zero_state(num_sites, chi)
        mps1, _ = hs.mps_evolution_order2(true_params, deltat, steps, mps)
        mps2, _ = hs.mps_evolution_order2(true_params, deltat, steps, mps1)
        mps3, _ = hs.mps_evolution_order2(true_params, deltat, steps, mps2)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps1) - states[0]), 0., delta=1e-4)    
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps2) - states[1]), 0., delta=1e-4)
        self.assertAlmostEqual(np.linalg.norm(mps_to_vector(mps3) - states[2]), 0., delta=1e-4)
    
    # def test_heisenberg_disordered_grad(self):