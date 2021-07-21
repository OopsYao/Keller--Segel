import numpy as np
import solver.utils as utils


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


M = 45
N = 50
m0 = 1
w0 = np.linspace(0.1, m0, M)
V0 = w0 / ((w0 + 0.01) * (1.01 - w0)) ** (1 / 4)
c0 = np.ones(N)
chi = 1
def R_rho(_): return 0
