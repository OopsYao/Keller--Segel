from .init_diff import Phi0, u0
import fenics as fen
import solver.fen.utils as utl
import ufl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Base setup
V = Phi0.function_space()
V0 = u0.function_space()
mesh = V.mesh()
dim = mesh.coordinates().shape[-1]
one_dim = dim == 1

# Do not know why, but it works
# https://fenicsproject.org/qa/12715/error-with-interpolate-after-using-ale-move/
mesh.bounding_box_tree().build(mesh)


class TermV:
    def __init__(self, v):
        VV = fen.VectorFunctionSpace(mesh, 'P', 1)
        VVV = fen.TensorFunctionSpace(mesh, 'P', 1, shape=(dim, dim))
        self._v = v
        grad = fen.grad(self._v)
        hess = fen.grad(grad)
        # The projection is necessary due to the composition
        # (the outer function value must be evaluated)
        self._grad = fen.project(grad, VV)
        self._hess = fen.project(hess, VVV)
        self._grad.set_allow_extrapolation(True)
        self._hess.set_allow_extrapolation(True)

    def F(self, u):
        v = fen.TestFunction(V)
        return fen.dot(utl.ComposeExpression(self._grad, u), v) * fen.dx

    def J(self, u):
        composed = utl.ComposeExpression(self._hess, u)
        v = fen.TestFunction(V)
        du = fen.TrialFunction(V)
        return fen.dot(composed * du, v) * fen.dx


def cof(u):
    v = fen.TestFunction(V)
    D = fen.grad(u)
    if dim == 1:
        # Cofactor_expr is not implemented for dimension 1
        form = fen.Identity(1) / 2 / ufl.det(D) ** 2
    else:
        form = ufl.transpose(ufl.cofac(D)) / 2 / ufl.det(D) ** 2
    return fen.inner(form, fen.grad(v)) * fen.dx


def recover(Phi):
    '''Recovers rho (composed Phi) with given Phi'''
    epsilon = 1e-8
    V = fen.FunctionSpace(mesh, 'P', 4)
    # return fen.project(1 / ufl.det(fen.grad(Phi)), V)
    u = fen.TrialFunction(V)
    v = fen.TestFunction(V)
    a = u * v * fen.dx - epsilon * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx
    f = 1 / ufl.det(fen.grad(Phi)) * v * fen.dx
    rho_Phi = fen.Function(V)
    fen.solve(a == f, rho_Phi)
    return rho_Phi


# An workaround for adaptive relaxation parameter
sld = False


def operator_T(Phi, v, dt):
    Phi_t = Phi
    u = Phi_t.copy(True)  # Use last time value as initial condition
    F = -fen.dot((u - Phi_t) / dt, fen.TestFunction(V)) * fen.dx \
        + cof(u)
    J = fen.derivative(F, u, fen.TrialFunction(V))

    if '-chi' not in sys.argv:
        chi = 17
        tv = TermV(v)
        u.set_allow_extrapolation(True)
        F = F + chi * tv.F(u)
        J = J + chi * tv.J(u)

    bcs = [fen.DirichletBC(u.function_space(),
                           fen.Expression(tuple(f'x[{i}]' for i in range(dim)),
                                          degree=1),
                           lambda x, on_bdy: on_bdy)]
    problem = fen.NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
    solver = fen.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-5
    prm['newton_solver']['relative_tolerance'] = 1e-4
    global sld
    prm['newton_solver']['relaxation_parameter'] = 0.3 if not sld else 0.8
    prm['newton_solver']['maximum_iterations'] = 250
    solver.solve()
    sld = True

    return u, v


def operator_S(Phi, v, dt):
    # Note here u is u composed with Phi indeed
    v_t = v
    v = fen.TrialFunction(V0)
    w = fen.TestFunction(V0)
    # Since u_Phi (u \circ Phi) is a function of V,
    # it is convenient to set v in the same space
    u_Phi = recover(Phi)
    u = extract_rho(u_Phi, x)
    u.set_allow_extrapolation(True)
    u = fen.interpolate(u, V0)
    F = -(v - v_t) / dt * w * fen.dx \
        - fen.dot(fen.grad(v), fen.grad(w)) * fen.dx \
        + (u - v) * w * fen.dx
    a, L = fen.lhs(F), fen.rhs(F)
    v = fen.Function(V0)
    fen.solve(a == L, v)
    return Phi, v


def extract_rho(rho_Phi, trans_coord):
    mesh = fen.Mesh(V.mesh())
    # One-dimension workaround
    if len(trans_coord.shape) == 1:
        trans_coord = np.expand_dims(trans_coord, -1)
    mesh.coordinates()[:] = trans_coord
    return utl.func_from_vertices(mesh, rho_Phi.compute_vertex_values())


def plot_mesh(Phi, title, filename=None):
    mesh = fen.Mesh(V.mesh())
    raw_x = Phi.compute_vertex_values()
    size = raw_x.size
    x = raw_x.reshape((2, int(size / 2))).T if dim != 1 else raw_x
    mesh.coordinates()[:] = x if not one_dim else np.expand_dims(x, -1)
    plt.figure(title)
    fen.plot(mesh)
    plt.savefig(
        f'artifacts/{dim}d_{title if filename is None else filename}.pdf')
    plt.close()


def my_plot(func, title, filename=None):
    plt.figure(title)
    c = fen.plot(func)
    if not one_dim:
        plt.colorbar(c)
    plt.savefig(
        f'artifacts/{dim}d_{title if filename is None else filename}.pdf')
    plt.close()


with open(f'x_dim={dim}.npy', 'rb') as f:
    x = np.load(f)
# Given Phi, v
dt = 0.01
Phi = Phi0
v = fen.interpolate(fen.Expression('pow(x[0], 2)'
                                   if one_dim else
                                   'pow(x[0], 2) + pow(x[1], 2)',
                                   degree=2), V0)
f_u = fen.File(f'artifacts/{dim}d_u.pvd')
f_v = fen.File(f'artifacts/{dim}d_v.pvd')
my_plot(v, 'v0')

t = 0
f_v << (v, t)
f_u << (u0, t)
my_plot(u0, 'u0')
for i in tqdm(range(2000)):
    t += dt
    Phi, v = operator_T(Phi, v, dt / 2)
    Phi, v = operator_S(Phi, v, dt)
    Phi, v = operator_T(Phi, v, dt / 2)
    f_v << (v, t)
    u_Phi = recover(Phi)
    u = extract_rho(u_Phi, x)
    f_u << (u, t)

    # Validate that det(D Phi) > 0
    assert u_Phi.compute_vertex_values().min() > 0
    assert u.compute_vertex_values().min() > 0
    Phi2 = fen.project(fen.dot(Phi, Phi), V0)
    values = Phi2.compute_vertex_values()
    m, M = values.min(), values.max()
    assert abs(M - 1) < 1e-2 or one_dim
    plot_mesh(Phi, f'Phi{i+1}')
    my_plot(u, f'u{i+1}')
    my_plot(v, f'v{i+1}')
