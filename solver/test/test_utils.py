import unittest
import solver.utils as utils
import numpy as np
import numpy.testing as npt


class TestInterpolationMethods(unittest.TestCase):

    def test_hat_interpolation(self):
        x = np.array([1, 2, 4])
        y = np.array([2, 1, 3])
        eta = np.array([1.1, 3.8, 1, 4])  # With a random order (not monotoned)
        npt.assert_array_equal(np.array([1.9, 2.8, 2, 3]),
                               utils.hat_interpolate(x, y, eta))

    def test_spline_interpolation_bdc(self):
        x = np.array([1, 2, 4, 5])
        y = np.array([2, 1, 3, -1])
        eta = np.array([5, 1])
        npt.assert_array_equal(np.array([0, 0]),
                               utils.spline_interpolate(x, y, eta, 1))
        npt.assert_array_equal(y, utils.spline_interpolate(x, y, x))

    def test_hat_inner_product(self):
        x = np.array([-1, 0, 1, 3])
        m = np.array([
            [1 / 2, 1 / 6, 0, 0],
            [1 / 6, 1, 1 / 6, 0],
            [0, 1 / 6, 3 / 2, 1 / 3],
            [0, 0, 1 / 3, 1]
        ])
        mm = np.array([
            [1, -1, 0, 0],
            [-1, 2, -1, 0],
            [0, -1, 3 / 2, -1/2],
            [0, 0, -1/2, 1/2]
        ])
        npt.assert_array_equal(m, utils.hat_inner_product(x))
        npt.assert_array_equal(mm, utils.hat_der_inner_product(x))


if __name__ == '__main__':
    unittest.main()
