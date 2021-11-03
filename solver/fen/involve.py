import fenics as fen
import solver.fen.utils as utl
import ufl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from .diffeo import post_process, pre_process_pse_inv
import solver.helper as helper

# Base setup
dim = 1
one_dim = dim == 1

# Do not know why, but it works
# https://fenicsproject.org/qa/12715/error-with-interpolate-after-using-ale-move/
# mesh.bounding_box_tree().build(mesh)


class TermV:
    def __init__(self, v):
        mesh = v.function_space().mesh()
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
        V = u.function_space()
        v = fen.TestFunction(V)
        return fen.dot(utl.ComposeExpression(self._grad, u), v) * fen.dx

    def J(self, u):
        V = u.function_space()
        composed = utl.ComposeExpression(self._hess, u)
        v = fen.TestFunction(V)
        du = fen.TrialFunction(V)
        return fen.dot(composed * du, v) * fen.dx


def cof(u):
    V = u.function_space()
    v = fen.TestFunction(V)
    D = fen.grad(u)
    if dim == 1:
        # Cofactor_expr is not implemented for dimension 1
        form = fen.Identity(1) / 2 / ufl.det(D) ** 2
    else:
        form = ufl.transpose(ufl.cofac(D)) / 2 / ufl.det(D) ** 2
    return fen.inner(form, fen.grad(v)) * fen.dx


# An workaround for adaptive relaxation parameter
sld = False


def operator_T(Phi, v, dt):
    Phi_t = Phi
    V = Phi.function_space()
    u = Phi_t.copy(True)  # Use last time value as initial condition
    F = -fen.dot((u - Phi_t) / dt, fen.TestFunction(V)) * fen.dx \
        + cof(u)
    J = fen.derivative(F, u, fen.TrialFunction(V))

    if '-chi' not in sys.argv:
        chi = 6
        tv = TermV(v)
        u.set_allow_extrapolation(True)
        F = F + chi * tv.F(u)
        J = J + chi * tv.J(u)

    if dim == 1:
        tol = 1e-4
        M = V.mesh().coordinates()[-1]
        a, b = Phi(0), Phi(M)
        bcs = [fen.DirichletBC(
            V, fen.Expression((f'{a}',), degree=1),
            lambda x, on_bdy: on_bdy and abs(x[0]) < tol),
            fen.DirichletBC(
            V, fen.Expression((f'{b}',), degree=1),
            lambda x, on_bdy: on_bdy and abs(x[0] - M) < tol),
        ]
    else:
        bcs = [fen.DirichletBC(
            u.function_space(),
            fen.Expression(tuple(f'x[{i}]' for i in range(dim)),
                           degree=1),
            lambda x, on_bdy: on_bdy)]
    problem = fen.NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
    solver = fen.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-5
    prm['newton_solver']['relative_tolerance'] = 1e-4
    global sld
    prm['newton_solver']['relaxation_parameter'] = 0.3 if not sld else 1.0
    prm['newton_solver']['maximum_iterations'] = 500
    solver.solve()
    sld = True

    return u, v


def operator_S(Phi, v, dt):
    V = v.function_space()
    u = post_process(Phi)
    u.set_allow_extrapolation(True)
    u = fen.interpolate(u, V)
    v_t = v
    v = fen.TrialFunction(V)
    w = fen.TestFunction(V)
    F = -(v - v_t) / dt * w * fen.dx \
        - fen.dot(fen.grad(v), fen.grad(w)) * fen.dx \
        + (u - v) * w * fen.dx
    a, L = fen.lhs(F), fen.rhs(F)
    v = fen.Function(V)
    fen.solve(a == L, v)
    return Phi, v


def plot_mesh(Phi, title, filename=None):
    mesh = fen.Mesh(Phi.function_space().mesh())
    raw_x = Phi.compute_vertex_values()
    size = raw_x.size
    x = raw_x.reshape((2, int(size / 2))).T if dim != 1 else raw_x
    mesh.coordinates()[:] = x if not one_dim else np.expand_dims(x, -1)
    plt.figure(title)
    fen.plot(mesh)
    plt.savefig(
        f'artifacts/{dim}d_{title if filename is None else filename}.pdf')
    plt.close()


def my_plot(func, title, filename=None, marker=False):
    plt.figure(title)
    if marker:
        m = {'marker': 'x',  'markersize': 2}
    else:
        m = {}
    c = fen.plot(func, **m)
    if not one_dim:
        plt.colorbar(c)
    plt.savefig(
        f'{helper.artifacts_dir}/'
        f'{title if filename is None else filename}.pdf')
    plt.close()


def initial_value(fenics_func=True):
    mesh = fen.IntervalMesh(1000, 0, np.pi)  # Mesh for v
    V = fen.FunctionSpace(mesh, 'P', 4)
    if fenics_func:
        u = fen.Expression('(x[0] >= pi/2 - 1) && (x[0] <= pi/2 + 1) ?'
                           '3.0/4 * (1 - pow(x[0] - pi/2, 2)) : 0',
                           degree=2, pi=np.pi)
        u = fen.interpolate(u, V)
    else:
        def u_func(x):
            return np.where(
                (x > np.pi/2 - 1) & (x < np.pi/2 + 1),
                3/4 * (1 - (x - np.pi/2) ** 2),
                np.zeros_like(x))
        u = {'func': u_func, 'a': 0, 'b': np.pi, 'N': 1000}
    v = fen.Expression(
        '1.2 * exp(-3 * pow(x[0], 2))'
        '+ 1.2 * exp(-3 * pow(x[0] - pi, 2))',
        degree=4, pi=np.pi)
    v = fen.interpolate(v, V)
    return u, v


# Initial values
u, v = initial_value(False)
Phi = pre_process_pse_inv(u)
dt = 0.01
t = 0
u = post_process(Phi, epsilon=0)
my_plot(u, 'raw-u0', marker=True)
u.set_allow_extrapolation(True)
u = fen.interpolate(u, v.function_space())
my_plot(Phi[0], 'Phi0', marker=True)
my_plot(u, 'u0', marker=True)
my_plot(v, 'v0', marker=False)
done = False
count = 0
for i in tqdm(range(10000)):
    t += dt
    Phi_last, v_last = Phi, v
    Phi, v = operator_T(Phi, v, dt / 2)
    Phi, v = operator_S(Phi, v, dt)
    Phi, v = operator_T(Phi, v, dt / 2)

    u = post_process(Phi, epsilon=0)
    my_plot(u, f'raw-u{i+1}', marker=True)
    u.set_allow_extrapolation(True)
    u = fen.interpolate(u, v.function_space())

    my_plot(Phi[0], f'Phi{i+1}', marker=True)
    my_plot(u, f'u{i+1}', marker=True)
    my_plot(v, f'v{i+1}', marker=False)

    # Terminal condition
    if not done:
        err_Phi = fen.errornorm(Phi_last, Phi, 'l2')
        err_v = fen.errornorm(v_last, v, 'l2')
        if err_Phi < 1e-9 and err_v < 1e-9:
            done = True
    else:
        count += 1
    if count == 1000:
        break
