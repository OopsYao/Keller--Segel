import numpy as np
from scipy import interpolate


def hat_interpolate(x, y, eta):
    '''Calculate the hat function interpolation at position eta
    based on the mesh x and corresponding value y'''
    # Right index of each surrounding interval
    r = np.minimum(x.shape[-1] - 1, np.digitize(eta, x))
    xr = x[r]
    xl = x[r - 1]
    yr = y[r]
    yl = y[r - 1]
    return ((xr - eta) * yl + (eta - xl) * yr) / (xr - xl)


def spline_interpolate(x, y, eta, der=0):
    '''Calculate the spline interpolation (or its derivative) at position eta'''
    cs = interpolate.CubicSpline(x, y, bc_type='clamped')
    return cs(eta, der)


def hat_inner_product(x):
    '''Calculate the inner products of hat functions on the mesh x'''
    d2x = x[2:] - x[:-2]
    main = np.abs([x[1] - x[0], *d2x, x[-1] - x[-2]]) / 2
    dx = x[1:] - x[:-1]
    sub = np.abs(dx) / 6
    return np.diag(main) + np.diag(sub, -1) + np.diag(sub, 1)


def hat_der_inner_product(x):
    '''Calculate the inner products of hat functions' derivatives on the mesh x'''
    dx = x[1:] - x[:-1]
    main = [1 / (x[1] - x[0]), *(1 / np.abs(dx[1:]) +
                                 1 / np.abs(dx[:-1])), 1 / (x[-1] - x[-2])]
    sub = -1 / np.abs(dx)
    return np.diag(main) + np.diag(sub, -1) + np.diag(sub, 1)
