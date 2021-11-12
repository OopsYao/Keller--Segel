import solver.fd.operators as fd
from solver.fd.spec import DiscreteFunc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import solver.context as ctx


def u0(x):
    return 3 / 4 * np.where(1 > (x - np.pi / 2) ** 2,
                            1 - (x - np.pi / 2) ** 2, 0)


def v0(x):
    return 1.2 * np.exp(-3 * x ** 2) \
        + 1.2 * np.exp(-3 * (x - np.pi) ** 2)


a = ctx.a
b = ctx.b
x = np.linspace(a, b, 100)
dt = 0.01
u = DiscreteFunc.equi_x(u0(x), a, b)
Phi = fd.pre_process(u, 100)
u_rec = fd.post_process(Phi)
v = DiscreteFunc.equi_x(v0(x), a, b)
plt.figure('u')
plt.plot(u.x, u.y, label='exact')
plt.plot(u_rec.x, u_rec.y)
plt.figure('v')
plt.plot(v.x, v.y)
plt.figure('Phi')
plt.plot(Phi.x, Phi.y)
with tqdm() as pbar:
    while True:
        Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
        u = fd.post_process(Phi)
        v = fd.implicit_v(v, u.interpolate('next'), dt)
        Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
        A, dv = fd.JF_S(v, u.interpolate('next'))
        _, dPhi = fd.JF_T(Phi, v.interpolate('spline'))
        r1, r2 = np.abs(A @ v.y + dv).max(), np.abs(dPhi).max()
        # Equilibrium
        if r1 < 1e-9 and r2 < 1e-9:
            break
        pbar.set_description(f'Steady: {r1:.2e}, {r2:.2e}')
        pbar.update(1)
plt.figure('Phi')
plt.plot(Phi.x, Phi.y)
plt.figure('u')
plt.plot(u.x, u.y)
plt.legend()
plt.figure('v')
plt.plot(v.x, v.y)
plt.show()
