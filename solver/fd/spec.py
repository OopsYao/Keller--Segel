from scipy.interpolate import CubicSpline
import numpy as np


class DiscreteFunc:
    def __init__(self, y, x_range):
        self.y = np.array(y)
        self.a = x_range[0]
        self.b = x_range[1]
        self.n = len(y)
        self.dx = (self.b - self.a) / self.n

    def __call__(self, x, der=0):
        '''Evaluate by interpolation'''
        cs = CubicSpline(np.linspace(self.a, self.b, self.n),
                         self.y, bc_type='clamped')
        return cs(x, der)

    def __add__(self, to_add):
        arr = self._extract_arr(to_add)
        return DiscreteFunc(self.y + arr, [self.a, self.b])

    def __sub__(self, to_sub):
        arr = self._extract_arr(to_sub)
        return DiscreteFunc(self.y - arr, [self.a, self.b])

    def __mul__(self, to_mul):
        arr = self._extract_arr(to_mul)
        return DiscreteFunc(self.y * arr, [self.a, self.b])

    def __truediv__(self, to_div):
        arr = self._extract_arr(to_div)
        return DiscreteFunc(self.y / arr, [self.a, self.b])

    def _extract_arr(self, func):
        # TODO Raise error if [a, b] does not match
        if isinstance(func, DiscreteFunc):
            arr = func.y
        elif isinstance(func, (list, tuple, np.ndarray)):
            arr = func
        return arr
