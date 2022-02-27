import numpy as np
import solver.context as ctx
import scipy.integrate as inte
from solver.fd.spec import DiscreteFunc, AnalyticFunc
from tqdm import tqdm


def diff_half(y, dx):
    return (y[1:] - y[:-1]) / dx


def diff_mid(y, dx):
    return (y[2:] - y[:-2]) / (2 * dx)


def D(n, dx):
    return (-np.diag(np.ones(n - 1), -1) + np.eye(n))[1:] / dx


def JF_T(Phi, v):
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
        J_int, F_int = JF_T(Phi, v)
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
        if np.abs(d_Phi.max()) < 1e-8:
            break
    # Correction, there may be computing error
    Phi.y[0] = Phi_t.y[0]
    Phi.y[-1] = Phi_t.y[-1]
    return Phi


def JF_S(v, u):
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

    return A, b


def implicit_v(v: DiscreteFunc, u: callable, dt):
    n = v.n

    # RHS
    A, b = JF_S(v, u)

    # LHS
    A = A - np.identity(n) / dt
    b = b + v.y / dt

    y = np.linalg.solve(A, -b)
    return DiscreteFunc.equi_x(y, v.a, v.b)


def pre_process(rho: AnalyticFunc, n) -> DiscreteFunc:
    '''Convert a function to the its pseudo inverse distribution.
    Here n is the number of discrete points of the pseudo inverse distribution.'''
    a = rho.a
    b = rho.b

    M = inte.quad(rho, a, b)[0]
    # Potential inital points set. Too dense points are bad too.
    x = np.linspace(a, b, max(100, 2 * n))
    supp = x[rho(x) > 1e-7]  # Support of rho
    Phi_list = [a]

    def Rho(upper, div=1):
        r, err = inte.quad(lambda x: rho(x) / div, a, upper)
        return r

    def nearest(p, points):
        n = points[0]
        for pp in points[1:]:
            if abs(p - pp) < abs(p - n):
                n = pp
        return n

    for x_tilde in tqdm(np.linspace(0, M, n)[1:-1]):
        # Solve Phi
        # Initial guess is near the last solution.
        x = supp[supp >= Phi_list[-1]][0]
        count = 0
        while True:
            count = count + 1
            # Correcting to nearest supporting points
            if rho(x) < 1e-10:
                x = max(nearest(x, supp), Phi_list[-1])
            dx = -(Rho(x) - x_tilde) / rho(x)
            # If trapped, decrease step size
            # if count > 500:
            #     dx = dx / 3
            # elif count > 1000:
            #     dx = dx / 10
            x = x + dx
            print(abs(Rho(x) - x_tilde), rho(x), dx)

            if abs(Rho(x) - x_tilde) < 1e-9:
                Phi_list.append(x)
                break
    Phi_list.append(b)
    return DiscreteFunc.equi_x(Phi_list, 0, M)


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


def CFL(Phi: DiscreteFunc, v: DiscreteFunc):
    p1 = 1 / (ctx.chi * np.abs(v.interpolate('spline')(Phi.y, 2)).max())
    p2 = 100 * v.dx
    return 0.3 * min(p1, p2)


def free_energy(u, v):
    '''Free energy of u and v. Here u and v are (equidistant) DiscreteFunc
    and share the same x-axis'''
    dx = v.dx
    # Given that v holds the HNBC, then the derivatives of u at the boundary can
    # be safely omitted.
    vx = (v.y[1:] - v.y[:-1]) / dx
    p1 = (u.y ** 2).sum() * dx / ctx.chi
    p2 = ((vx ** 2).sum() + (v.y ** 2 - 2 * u.y * v.y).sum()) * dx
    return p1 + p2


def mass(u):
    '''Mass of the function. Here u is a (equidistant) DiscreteFunc'''
    return u.y.sum() * u.dx
