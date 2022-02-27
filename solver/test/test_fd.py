import unittest
from solver.fd.operators import diff_half, D, pre_process, post_process
from solver.fd.spec import AnalyticFunc, DiscreteFunc
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

    def test_pre_process(self):
        rho = AnalyticFunc(lambda x: x ** 2, 0, 1)
        for n in [2, 3, 50, 100, 199, 200, 201, 500, 1000, 5000]:
            Phi = pre_process(rho, n)
            err = (Phi.y - (3 * Phi.x) ** (1 / 3)).max()
            self.assertAlmostEqual(err, 0)

        # Try to transfer a function with compact support
        def rho(x):
            return 3 / 4 * np.where(1 > (x - np.pi / 2) ** 2,
                                    1 - (x - np.pi / 2) ** 2, 0)
        rho = AnalyticFunc(rho, 0, np.pi)
        Phi = pre_process(rho, 500)

    def test_post_process(self):
        x = np.linspace(0, 1 / 3, 500)
        Phi = DiscreteFunc.equi_x((3 * x) ** (1 / 3), 0, 1 / 3)
        rho = post_process(Phi)
        err = (rho.y - rho.x ** 2).max()
        self.assertAlmostEqual(err, 0, 1)
