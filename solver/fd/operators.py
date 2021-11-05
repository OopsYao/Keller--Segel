import numpy as np
import solver.context as ctx


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


def implicit(Phi, v, dt):
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
    return Phi
