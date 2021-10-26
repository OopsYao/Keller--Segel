import fenics as fen
import numpy as np
import ufl
from solver.fen.utils import func_from_vertices
import scipy.integrate as inte


def post_process(Phi, epsilon=1e-4, degree=4):
    '''Recovers rho from Phi'''
    mesh = Phi.function_space().mesh()
    V = fen.FunctionSpace(mesh, 'P', degree)  # Space of rho composing Phi

    # rho composing Phi
    if epsilon == 0:
        f = 1 / ufl.det(fen.grad(Phi))
        rho_Phi = fen.project(f, V)
    else:
        u = fen.TrialFunction(V)
        v = fen.TestFunction(V)
        a = u * v * fen.dx \
            - epsilon * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx
        f = 1 / ufl.det(fen.grad(Phi)) * v * fen.dx
        rho_Phi = fen.Function(V)
        fen.solve(a == f, rho_Phi)

    # Copy function rho_Phi (to rho) and its space
    mesh_mirror = fen.Mesh(mesh)  # This copys mesh coordinates
    V_mirror = fen.FunctionSpace(mesh_mirror, 'P', degree)
    rho = fen.Function(V_mirror)
    rho.vector()[:] = np.array(rho_Phi.vector())

    # Now we apply the mesh transform to recover rho from rho composing Phi.
    # The domain mesh of rho is not equidistant any more.
    M, d = mesh_mirror.coordinates().shape
    trans_coord = np.array(Phi.compute_vertex_values()).reshape((d, M)).T
    mesh_mirror.coordinates()[:] = trans_coord
    return rho


def pre_process(rho):
    '''Convert rho to Phi'''

    def heat_equation(rho0, dt, t0=0):
        V0 = rho0.function_space()
        u_n = rho0.copy(True)
        u = fen.TrialFunction(V0)
        v = fen.TestFunction(V0)

        F = u * v * fen.dx + dt * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx \
            - u_n * v * fen.dx
        a, L = fen.lhs(F), fen.rhs(F)

        t = t0
        rho = fen.Function(V0)
        while True:
            t += dt
            fen.solve(a == L, rho)
            l2 = fen.errornorm(rho, u_n, 'l2')
            yield rho.copy(True), t, l2
            u_n.assign(rho)  # Update a, L

    def ODE(rho_list, x_inf, dt):
        mesh = rho_list[0].function_space().mesh()
        VV = fen.VectorFunctionSpace(mesh, 'P', 2)
        x_list = []
        x = x_inf
        for rho in reversed(rho_list):
            v = fen.project(-fen.grad(rho) / rho, VV)
            v.set_allow_extrapolation(True)
            dnc = dt * np.array([v(p) for p in x])
            x_list.append(x)
            x = x - dnc
        x_list = np.flip(np.array(x_list), 0)
        return x_list

    mesh = fen.Mesh(rho.function_space().mesh())
    rho_list = [rho]
    dt = 0.01
    for rho, _, l2 in heat_equation(rho, dt):
        rho_list.append(rho)
        if l2 < 1e-6:
            break
    x = ODE(rho_list, np.squeeze(mesh.coordinates()), dt)[0]
    Phi = func_from_vertices(mesh, x, squeeze=False)
    return Phi


def pre_process_pse_inv(rho):
    a = rho['a']
    b = rho['b']
    N = rho['N']  # Number of cells
    rho = rho['func']

    M = inte.quad(rho, a, b)[0]
    # Omega_tilde := [0, M] (equidistant)
    x_tilde = np.linspace(0, M, N + 1)
    # Initial guess
    x = np.linspace(a, b, N + 1)

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
    mesh = fen.IntervalMesh(N, a, b)
    Phi = func_from_vertices(mesh, x, squeeze=False)
    return Phi
