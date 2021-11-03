import fenics as fen
import matplotlib.pyplot as plt
import numpy as np
import ufl
from solver.fen.utils import func_from_vertices, ComposeExpression
from tqdm import tqdm
import mshr as mh
import sys

dt = 0.001

# fen.set_log_level(30)  # Only error
# Create mesh and define function space
one_dim = ','.join(['-d', '1']) in ','.join(sys.argv)
if one_dim:
    mesh = fen.IntervalMesh(100, 0, np.pi)
else:
    domain = mh.Circle(fen.Point(0, 0), 1)
    mesh = mh.generate_mesh(domain, 30)
dim = mesh.coordinates().shape[-1]
plt.figure('comp-mesh')
fen.plot(mesh)
plt.savefig('artifacts/comp-mesh.pdf')
V0 = fen.FunctionSpace(mesh, 'P', 2)  # Function space of u, v
V = fen.VectorFunctionSpace(mesh, 'P', 1)  # Function space of Phi


def make_u_0():
    # Adjust constant c to make u0 to have the same integral with constant 1 on
    # the domain.
    sigma = 0.3
    params = {'degree': 6, 'sigma2': sigma ** 2, 'pi': np.pi}
    signature = 'x[0] >= pi/2 - 1 && x[0] <= pi/2 + 1 ?' \
        '3/4 * (1 - pow(x[0] - pi/2, 2)) : 0' \
        if one_dim else \
        'exp(-(pow(x[0], 2) + pow(x[1], 2)) / 2 / sigma2) / 2 / sigma2 / pi'
    main_expr = fen.Expression(signature, **params)
    main = fen.interpolate(main_expr, V0)
    unit = fen.project(1, V0)
    c = 1 - fen.assemble(main * fen.dx) / fen.assemble(unit * fen.dx)
    return fen.interpolate(
        fen.Expression(signature + '+ c', c=c, **params),
        V0)


def heat_equation(rho0, dt, t0=0):
    u_n = rho0.copy(True)
    u = fen.TrialFunction(V0)
    v = fen.TestFunction(V0)

    F = u * v * fen.dx + dt * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx - \
        u_n * v * fen.dx
    a, L = fen.lhs(F), fen.rhs(F)

    t = t0
    rho = fen.Function(V0)
    while True:
        t += dt
        fen.solve(a == L, rho)
        # Save to file and plot solution
        l2 = fen.errornorm(rho, u_n, 'L2')
        yield rho.copy(True), t, l2

        # Update a, L
        u_n.assign(rho)


# Solve the ODE of x(t)
def ODE(rho_list, x_inf):
    VV = fen.VectorFunctionSpace(mesh, 'P', 2)
    x_list = []
    x = x_inf.copy()
    for rho in tqdm([*reversed(rho_list)]):
        v = fen.project(-fen.grad(rho) / rho, VV)
        v.set_allow_extrapolation(True)
        dnc = dt * np.array([v(p) for p in x])
        x_list.append(x)
        x = x - dnc
    x_list = np.flip(np.array(x_list), 0)
    return x_list


# Validate Phi0
rho0 = make_u_0()
u0 = rho0
try:
    x = np.load(f'x_dim={dim}.npy')
except Exception:
    # Solve heat equation
    heat_file = fen.File('artifacts/heat-rho.pvd')
    rho_list = [rho0]
    for rho, t, l2 in heat_equation(rho0, dt):
        rho_list.append(rho)
        heat_file << (rho, t)
        if l2 < 1e-6:
            break
    x = ODE(rho_list, np.squeeze(mesh.coordinates()))[0]
    with open(f'x_dim={dim}.npy', 'wb') as f:
        np.save(f, x)
Phi0 = func_from_vertices(mesh, x, squeeze=False)
rho0.set_allow_extrapolation(True)
Phi0.set_allow_extrapolation(True)
composed = ComposeExpression(rho0, Phi0)
d = ufl.det(fen.grad(Phi0))
err = fen.project(composed * d - 1, V0)
plt.figure('err')
fen.plot(err)
print('Err (L inf):', np.abs(err.compute_vertex_values(mesh)).max())
print('Err (L2):', fen.errornorm(err, fen.project(fen.Constant(0), V0), 'L2'))
Phi0 = fen.interpolate(Phi0, V)


def transform_mesh(mesh, x):
    trans_mesh = fen.Mesh(mesh)
    trans_mesh.coordinates()[:] = np.expand_dims(x, -1) if dim == 1 else x


# Hold plot
plt.show()
