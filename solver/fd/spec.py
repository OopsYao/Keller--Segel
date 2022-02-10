from scipy.interpolate import CubicSpline, interp1d
import numpy as np


class DiscreteFunc:
    def __init__(self, x, y):
        self.y = np.array(y)
        self.x = np.array(x)
        self.a = x[0]
        self.b = x[-1]
        self.n = len(x)
        self.dx = None

    @classmethod
    def equi_x(cls, y, a, b):
        n = len(y)
        func = cls(np.linspace(a, b, n), y)
        func.dx = (b - a) / (n - 1)
        return func

    def __call__(self, x, der=0):
        '''Evaluate by interpolation'''
        cs = CubicSpline(self.x, self.y, bc_type='clamped')
        return cs(x, der)

    def interpolate(self, kind='linear'):
        if kind == 'spline':
            cs = CubicSpline(self.x, self.y, bc_type='clamped')
            return cs
        elif kind == 'next':
            return interp1d(self.x, self.y, kind='next')
        else:
            return interp1d(self.x, self.y, kind='linear')

    def __add__(self, to_add):
        arr = self._extract_arr(to_add)
        return DiscreteFunc.equi_x(self.y + arr, self.a, self.b)

    def __sub__(self, to_sub):
        arr = self._extract_arr(to_sub)
        return DiscreteFunc.equi_x(self.y - arr, self.a, self.b)

    def __mul__(self, to_mul):
        arr = self._extract_arr(to_mul)
        return DiscreteFunc.equi_x(self.y * arr, self.a, self.b)

    def __truediv__(self, to_div):
        arr = self._extract_arr(to_div)
        return DiscreteFunc.equi_x(self.y / arr, self.a, self.b)

    def _extract_arr(self, func):
        # TODO Raise error if [a, b] does not match
        if isinstance(func, DiscreteFunc):
            arr = func.y
        elif isinstance(func, (list, tuple, np.ndarray, float, int)):
            arr = func
        return arr

    def norm(self, kind='L2'):
        if kind == 'L2':
            return np.sqrt(np.sum(self.y ** 2))
        elif kind == 'L1':
            return np.sum(np.abs(self.y))
        elif kind == 'Linf':
            return np.max(np.abs(self.y))
        else:
            raise ValueError('Unknown norm kind')


class AnalyticFunc:
    def __init__(self, func, a, b):
        self.func = func
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.func(x)
