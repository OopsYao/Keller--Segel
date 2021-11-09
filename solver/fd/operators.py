import numpy as np
import solver.context as ctx
import scipy.integrate as inte
from solver.fd.spec import DiscreteFunc, AnalyticFunc


def diff_half(y, dx):
    return (y[1:] - y[:-1]) / dx


def diff_mid(y, dx):
    return (y[2:] - y[:-2]) / (2 * dx)


def D(n, dx):
    return (-np.diag(np.ones(n - 1), -1) + np.eye(n))[1:] / dx


def JF(Phi, v):
    '''The discrete form of the interior J and F on the RHS'''
    dx = Phi.dx
    n = Phi.n
    Phi = Phi.y

    def op_dh(y):
        return diff_half(y, dx)

    def D_h(n):
        return D(n, dx)
    J = D_h(n - 1) @ np.diag(1 / op_dh(Phi) ** 3) @ D_h(n)
    F = -op_dh(1 / op_dh(Phi) ** 2) / 2

    J = J + ctx.chi * np.diag(v(Phi, 2))[1:-1]
    F = F + ctx.chi * v(Phi, 1)[1:-1]

    return J, F


def implicit_Phi(Phi: DiscreteFunc, v: callable, dt):
    '''Involve Phi by dt via implicit method'''
    Phi_t = Phi
    while True:
        J_int, F_int = JF(Phi, v)
        # Only F_int relies on Phi_t
        F_int = F_int - (Phi.y - Phi_t.y)[1:-1] / dt
        J_int = J_int - np.eye(Phi.n)[1:-1] / dt

        # d_Phi satisfies homogenous DBC as Phi satisfies a DBC
        F = np.zeros(Phi.n)
        J = np.eye(Phi.n)
        F[1:-1] = F_int
        J[1:-1] = J_int

        d_Phi = np.linalg.solve(J, -F)
        Phi = Phi + d_Phi
        if d_Phi.max() < 1e-8:
            break
    # Correction, there may be computing error
    Phi.y[0] = Phi_t.y[0]
    Phi.y[-1] = Phi_t.y[-1]
    return Phi


def implicit_v(v: DiscreteFunc, u: callable, dt):
    n = v.n

    # Laplace operator
    # The boundary value is computed by the homogenous NBC
    top = np.ones(n - 1)
    top[0] = 2
    bottom = np.ones(n - 1)
    bottom[-1] = 2
    mid = -2 * np.ones(n)
    laplace = (np.diag(top, 1) + np.diag(mid) + np.diag(bottom, -1)) \
        / (v.dx ** 2)

    # RHS
    A = laplace - np.identity(n)
    b = u(v.x)

    # LHS
    A = A - np.identity(n) / dt
    b = b + v.y / dt

    y = np.linalg.solve(A, -b)
    return DiscreteFunc.equi_x(y, v.a, v.b)


def pre_process(rho: AnalyticFunc, n) -> DiscreteFunc:
    # Here rho is an AnalyticFunc
    a = rho.a
    b = rho.b

    M = inte.quad(rho, a, b)[0]
    # Omega_tilde := [0, M] (equidistant)
    x_tilde = np.linspace(0, M, n)
    # Initial guess
    x = np.linspace(a, b, n)

    @np.vectorize
    def Rho(upper):
        return inte.quad(rho, a, upper)[0]

    unfulfill = np.full_like(x, True, dtype=bool)
    x[0], x[-1] = a, b
    unfulfill[0], unfulfill[-1] = False, False
    while unfulfill.any():
        # Correction
        invalid = (x < a) | (b < x)
        x[invalid] = np.random.uniform(a, b, invalid.sum())
        # Increment (where unfulfilled)
        inc = - (Rho(x[unfulfill]) -
                 x_tilde[unfulfill]) / rho(x[unfulfill])
        x[unfulfill] += inc
        unfulfill[unfulfill] = (np.abs(inc) >= 10e-8)
    return DiscreteFunc.equi_x(x, 0, M)


def post_process(Phi: DiscreteFunc) -> DiscreteFunc:
    dx = Phi.dx  # TODO Raise error if it is not equidistant
    n = Phi.n
    inter = 2 * dx / (Phi.y[2:] - Phi.y[:-2])
    rho_Phi = np.empty(n)  # rho(Phi) = 1 / D(Phi)
    rho_Phi[1:-1] = inter
    # Since rho satisfies HNBC, then Phi'' = 0 at the boundary
    # (if Phi' != inf), which gives a way to calculate Phi' there.
    # rho = 0 if Phi' = inf at the boundary.
    rho_Phi[0] = dx / (Phi.y[1] - Phi.y[0])
    rho_Phi[-1] = dx / (Phi.y[-1] - Phi.y[-2])
    return DiscreteFunc(Phi.y, rho_Phi)
