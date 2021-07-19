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


def CDC(v, system):
    r = system.get_c_mesh()
    M = utils.hat_inner_product(r)
    m = utils.hat_der_inner_product(r)
    N = 1  # TODO <phi' / r, phi>
    return -m + N - M


def non_linear(v, rho, system):
    r = system.get_c_mesh()
    dr = r[1] - r[0]
    return dr * utils.hat_interpolate(system.get_x(), rho, r)
