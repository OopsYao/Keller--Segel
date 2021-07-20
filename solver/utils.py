import numpy as np
from scipy import interpolate


def mask(V, mask_value):
    V_ij = np.expand_dims(V, -1) - V
    mask = np.abs(V_ij) >= 1e-6
    tran_V_ij = np.where(mask, V_ij, mask_value)
    return tran_V_ij


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


def hat_cross_inner_product(x):
    '''Cross inner products of phi' / x and phi'''
    dx = x[1:] - x[:-1]
    dlnx = np.log(x[1:] / x[:-1])

    a = (np.abs(dx) - x[1:] * np.abs(dlnx)) / dx ** 2
    b = (np.abs(dx) - x[:-1] * np.abs(dlnx)) / dx ** 2
    main = [a[0], *(b[:-1] + a[1:]), b[-1]]
    sub_up = - (dx - x[:-1] * dlnx) / dx ** 2
    sub_down = - (dx - x[1:] * dlnx) / dx ** 2
    return np.diag(main) + np.diag(sub_up, 1) + np.diag(sub_down, -1)


def matrix_elephant(dV_inv, dw, n):
    '''Numerial expression (matrix) of operater: W -> (W' / V'^n)'

        dV_inv -- 1 / np.diff(V)
        dw -- size of the corresponding mesh
        n -- power
    '''
    dV_p = np.array([*dV_inv, 0])
    dV_m = np.array([0, *dV_inv])

    main = dV_p ** n + dV_m ** n
    sub = dV_inv ** n
    return dw ** (n - 2) * (np.diag(sub, -1) - np.diag(main) + np.diag(sub, 1))


def V_rooster(dV_inv, dw, n):
    '''Numerial expression (vector) of (1 / V'^n)'

        dV_inv -- 1 / np.diff(V)
        dw -- size of the corresponding mesh
        n -- power
    '''
    dV_p = np.array([*dV_inv, 0])
    dV_m = np.array([0, *dV_inv])
    return dw ** (n - 1) * (dV_p ** n - dV_m ** n)
