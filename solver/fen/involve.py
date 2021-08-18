import fenics as fen
import mshr as mh
import solver.fen.utils as utl
import ufl
import numpy as np
import matplotlib.pyplot as plt

# Base setup
degree = 1
domain = mh.Circle(fen.Point(0, 0), 1)
mesh = mh.generate_mesh(domain, 30)
V = fen.FunctionSpace(mesh, 'P', degree)
VV = fen.VectorFunctionSpace(mesh, 'P', degree)
VVV = fen.TensorFunctionSpace(mesh, 'P', degree, shape=(2, 2))


class TermV:
    def __init__(self, v):
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
        v = fen.TestFunction(VV)
        return fen.dot(utl.ComposeExpression(self._grad, u), v) * fen.dx

    def J(self, u):
        composed = utl.ComposeExpression(self._hess, u)
        v = fen.TestFunction(VV)
        du = fen.TrialFunction(VV)
        return fen.dot(composed * du, v) * fen.dx


def cof(u):
    v = fen.TestFunction(VV)
    D = fen.grad(u)
    form = ufl.transpose(ufl.cofac(D)) / 2 / ufl.det(D) ** 2
    return fen.inner(form, fen.grad(v)) * fen.dx


def recover(Phi):
    '''Recovers rho (composed Phi) with given Phi'''
    epsilon = 1e-2
    u = fen.TrialFunction(V)
    v = fen.TestFunction(V)
    a = u * v * fen.dx - epsilon * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx
    f = 1 / ufl.det(fen.grad(Phi)) * v * fen.dx
    rho_Phi = fen.Function(V)
    fen.solve(a == f, rho_Phi)
    return rho_Phi


def operator_T(Phi, v, dt):
    Phi_t = Phi
    u = Phi_t.copy(True)  # Use last time value as initial condition
    F = -fen.dot((u - Phi_t) / dt, fen.TestFunction(VV)) * fen.dx + cof(u)
    J = fen.derivative(F, u, fen.TrialFunction(VV))

    tv = TermV(v)
    chi = 1
    J += chi * tv.J(u)
    F += chi * tv.F(u)

    problem = fen.NonlinearVariationalProblem(F, u, J=J)
    solver = fen.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-4
    prm['newton_solver']['relative_tolerance'] = 1e-3
    solver.solve()
    return u, v


def operator_S(Phi, v, dt):
    # Note here u is u composed with Phi indeed
    v_t = v
    v = fen.TrialFunction(V)
    w = fen.TestFunction(V)
    # Since rho_Phi is a function of V,
    # it is convenient to set v in the same space
    rho_Phi = recover(Phi)
    F = -(v - v_t) / dt * w * fen.dx \
        - fen.dot(fen.grad(v), fen.grad(w)) * fen.dx \
        + (rho_Phi - v) * w * fen.dx
    a, L = fen.lhs(F), fen.rhs(F)
    v = fen.Function(V)
    fen.solve(a == L, v)
    return Phi, v


def extract_rho(rho_Phi, trans_coord):
    mesh = mh.generate_mesh(domain, 30)
    mesh.coordinates()[:] = np.array(trans_coord)
    return utl.func_from_vertices(mesh, rho_Phi.compute_vertex_values())


with open('x.npy', 'rb') as f:
    x = np.load(f)
# Given Phi, v
dt = 0.01
Phi = utl.func_from_vertices(mesh, x)
v = fen.interpolate(fen.Expression(('(pow(x[0], 2) + pow(x[1], 2))'),
                                   degree=2), V)

f_rho = fen.File('artifacts/rho.pvd')
f_v = fen.File('artifacts/v.pvd')

plt.figure('rho0')
sigma = 0.3
c = np.exp(-1 / 2 / sigma ** 2) / np.pi + (np.pi - 1) / np.pi
u_0 = fen.Expression(
    'exp(-(pow(x[0], 2) + pow(x[1], 2))/2/pow(sigma, 2))/2/pow(sigma, 2)/pi'
    '+c',
    degree=6, sigma=sigma, pi=np.pi, c=c)
rho0 = fen.interpolate(u_0, V)
fen.plot(rho0)
plt.savefig('artifacts/rho0.pdf')
plt.figure('v0')
fen.plot(v)
plt.savefig('artifacts/v0.pdf')

t = 0
f_rho << (rho0, t)
f_v << (v, t)
for i in range(100):
    t += dt
    Phi, v = operator_T(Phi, v, dt)
    Phi, v = operator_S(Phi, v, dt)
    f_v << (v, t)
    rho_Phi = recover(Phi)
    rho = extract_rho(rho_Phi, x)
    f_rho << (rho, t)
plt.figure('rho')
fen.plot(rho)
plt.savefig('artifacts/rhoT.pdf')
plt.figure('v')
fen.plot(v)
plt.savefig('artifacts/vT.pdf')
plt.show()
