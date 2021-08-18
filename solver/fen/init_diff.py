import fenics as fen
import matplotlib.pyplot as plt
import numpy as np
import mshr as mh
import ufl
from solver.fen.utils import func_from_vertices, ComposeExpression
from tqdm import tqdm

dt = 0.001

degree = 2
# Create mesh and define function space
domain = mh.Circle(fen.Point(0, 0), 1)
mesh = mh.generate_mesh(domain, 30)
plt.figure('comp-mesh')
fen.plot(mesh)
plt.savefig('artifacts/comp-mesh.pdf')
V = fen.FunctionSpace(mesh, 'P', degree)

# Define initial value
sigma = 0.3
c = np.exp(-1 / 2 / sigma ** 2) / np.pi + (np.pi - 1) / np.pi
u_0 = fen.Expression(
    'exp(-(pow(x[0], 2) + pow(x[1], 2))/2/pow(sigma, 2))/2/pow(sigma, 2)/pi'
    '+c',
    degree=6, sigma=sigma, pi=np.pi, c=c)
rho0 = fen.interpolate(u_0, V)
u_n = rho0.copy(True)

# Total mass
print('integral of rho0:', fen.assemble(rho0 * fen.dx))

# Define variational problem
u = fen.TrialFunction(V)
v = fen.TestFunction(V)

F = u * v * fen.dx + dt * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx - \
    u_n * v * fen.dx
a, L = fen.lhs(F), fen.rhs(F)

# Create VTK file for saving solution
vtkfile = fen.File('artifacts/solution.pvd')

# Time-stepping
u = fen.Function(V)
u_list = [u_n.copy(True)]
t = 0

while True:
    # Update current time
    t += dt
    # Compute solution
    fen.solve(a == L, u)
    # Save to file and plot solution
    vtkfile << (u, t)
    l2 = fen.errornorm(u, u_list[-1], 'L2')

    # Update previous solution
    u_n.assign(u)
    u_list.append(u.copy(True))

    if l2 < 1e-6:
        break

u_list[-1].set_allow_extrapolation(True)
print('Max/min value of rho at equilibrium',
      u_list[-1](0, 0), u_list[-1](0, -1))

x = mesh.coordinates()
VV = fen.VectorFunctionSpace(mesh, 'P', degree)
x_list = []
for rho in tqdm([*reversed(u_list)]):
    v = fen.project(-fen.grad(rho) / rho, VV)
    v.set_allow_extrapolation(True)
    dnc = dt * np.array([v(p) for p in x])
    x_list.append(x)
    x = x - dnc
x_list = np.flip(np.array(x_list), 0)

x = x_list[0]  # !important
plt.figure('scatter')
plt.scatter(*(x.T), s=1, color='blue', label='trans')
plt.scatter(*(x_list[-1].T), s=1, color='red', label='equal')
plt.legend()

# Validate Phi0
Phi0 = func_from_vertices(mesh, x)
rho0.set_allow_extrapolation(True)
Phi0.set_allow_extrapolation(True)
composed = ComposeExpression(rho0, Phi0)
d = ufl.det(fen.grad(Phi0))
err = fen.project(composed * d - 1, V)
plt.figure('err')
fen.plot(err)
print('Err (L inf):', np.abs(err.compute_vertex_values(mesh)).max())
print('Err (L2):', fen.errornorm(err, fen.project(fen.Constant(0), V), 'L2'))

with open('x.npy', 'wb') as f:
    np.save(f, x)
mesh.coordinates()[:] = x
plt.figure('trans-mesh')
fen.plot(mesh)
plt.savefig('artifacts/trans-mesh.pdf')

# Hold plot
plt.show()
