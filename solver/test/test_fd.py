import unittest
from solver.fd.operators import diff_half, D
import numpy.testing as npt
import numpy as np


class TestFd(unittest.TestCase):
    def test_diff_half(self):
        dx = 0.5
        y = np.array([1, 4, 3])
        r = diff_half(y, dx)
        npt.assert_almost_equal(r, [6, -2])

    def test_D(self):
        dx = 0.3
        npt.assert_almost_equal(
            np.array([[-1, 1, 0],
                      [0, -1, 1]]) / dx,
            D(3, 0.3))
