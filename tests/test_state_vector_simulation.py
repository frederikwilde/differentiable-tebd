import unittest
import numpy as np
from differentiable_tebd.state_vector_simulation import vec_simulation, mps_to_vector
from differentiable_tebd.utils import X, Y, Z

class TestVecSimulation(unittest.TestCase):
    def setUp(self):
        self.hamiltonian = np.kron(Y, Z) + np.kron(np.eye(2), X)
        self.state = np.array([1, 0, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        self.times = [1., 1.5]
        self.result_zero_state = np.array([[ 0.15591746+0.j        , -0.52333301+0.j        ],
                                           [ 0.        -0.69858758j,  0.        -0.60251818j],
                                           [ 0.69858758+0.j        ,  0.60251818+0.j        ],
                                           [ 0.        +0.j        ,  0.        +0.j        ]])
        self.result_other_state = np.array([[ 0.11023312-0.49403951j, -0.37007074-0.42603308j],
                                            [ 0.        -0.49403951j,  0.        -0.42603308j],
                                            [ 0.49403951+0.11023312j,  0.42603308-0.37007074j],
                                            [ 0.49403951+0.j        ,  0.42603308+0.j        ]])
    
    def test_vec_simulation(self):
        v = vec_simulation(self.hamiltonian, self.times)
        np.testing.assert_array_almost_equal(v, self.result_zero_state)

    def test_initial_state(self):
        v = vec_simulation(self.hamiltonian, self.times, self.state)
        np.testing.assert_array_almost_equal(v, self.result_other_state)

class TestMPSToVector(unittest.TestCase):
    def setUp(self):
        self.mps = np.zeros((4, 3, 2, 3), dtype=np.complex128)
        for i in range(self.mps.shape[0]-1):
            self.mps[i, 0, 0, 0] = 1.
        self.mps[-1, 0, 1, 0] = 1.
        self.vec = np.zeros(2 ** 4, dtype=np.complex128)
        self.vec[1] = 1.

    def test_mps_to_vector(self):
        v = mps_to_vector(self.mps)
        np.testing.assert_array_almost_equal(v, self.vec)