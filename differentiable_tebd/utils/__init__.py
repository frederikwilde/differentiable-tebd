import numpy as np
from .. import COMPLEX_TYPE


X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX_TYPE)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)
Hadamard = np.array([[1, 1], [1, -1]], dtype=COMPLEX_TYPE) / np.sqrt(2.)
YHadamard = np.array([[1, -1j], [1, 1j]], dtype=COMPLEX_TYPE) / np.sqrt(2.)
