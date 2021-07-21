import matplotlib.pyplot as plt
import numpy as np
import solver.utils as utils
import scipy.special as ssp
from scipy.optimize import fsolve
from scipy.integrate import quad


def TDT(V, system):
    dw = system.get_dw()

    dV = np.diff(V)
    V2 = -utils.V_rooster(dV, dw, 2) / 2
    vp = utils.spline_interpolate(system.get_c_mesh(), system.c, V, 1)
    vpp = utils.spline_interpolate(system.get_c_mesh(), system.c, V, 2)

    chi = 1
    TV = V * (V2 + chi * vp)
    DT = np.diag(V) @ utils.matrix_elephant(dV, dw, 3) + \
        np.diag(V2 + chi * (V * vpp + vp))
    return TV, DT


def CDC(_, system):
    r = system.get_c_mesh()
    M = utils.hat_inner_product(r)
    m = utils.hat_der_inner_product(r)
    N = utils.hat_cross_inner_product(r)
    return -m + N - M


def non_linear(_, rho, system):
    r = system.get_c_mesh()
    dr = r[1] - r[0]
    return dr * utils.hat_interpolate(system.get_x(), rho, r)


def R_rho(_): return 0


M = 45
N = 50
m0 = 1
R = 1
chi = 1 + 3.84 ** 2 / R + 0.1
w = np.sqrt(chi - 1)


def r1_func(r1):
    return w * ssp.j0(w * r1) * (ssp.k1(R) * ssp.i1(r1) - ssp.i1(R) * ssp.k1(r1)) - \
        ssp.j1(w * r1) * (ssp.k1(R) * ssp.i0(r1) + ssp.i1(R) * ssp.k0(r1))


r1, *_ = fsolve(r1_func, R / 2)
A1 = 1 / (np.pi * r1 ** 2 * ssp.jv(2, w * r1))
B1 = - w ** 2 * ssp.j0(w * r1) / \
    (chi * np.pi * r1 ** 2 * ssp.jv(2, w * r1)) / \
    (ssp.k1(R) * ssp.i0(r1) + ssp.i1(R) * ssp.k0(r1))


def u0(r):
    epsilon = 0.01
    case1 = A1 * (ssp.j0(w * r) - ssp.j0(w * r1)) + epsilon
    case2 = epsilon
    return np.where(r < r1, case1, case2)


def v0(r):
    epsilon = 0.01
    case1 = A1 * (ssp.j0(w * r) / chi - ssp.j0(w * r1)) + epsilon
    case2 = B1 * (ssp.k1(R) * ssp.i0(r) + ssp.i1(R) * ssp.k0(r)) + epsilon
    return np.where(r < r1, case1, case2)


V0 = utils.inv_dist(u0, M, 0.1, R)  # TODO: The exact lower bound is 0 indeed
c0 = v0(np.linspace(V0[0], V0[-1], N))
m0, *_ = quad(u0, 0, R)
