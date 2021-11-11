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
        + 1 * np.exp(-3 * (x - np.pi) ** 2)


a = ctx.a
b = ctx.b
x = np.linspace(a, b, 100)
dt = 0.001
u = DiscreteFunc.equi_x(u0(x), a, b)
Phi = fd.pre_process(u, 100)
v = DiscreteFunc.equi_x(v0(x), a, b)
plt.plot(u.x, u.y)
plt.plot(v.x, v.y)
hist = []
for i in tqdm(range(30000)):
    Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
    u = fd.post_process(Phi)
    v = fd.implicit_v(v, u.interpolate('next'), dt)
    Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
    # Equilibrium
    if len(hist) >= 1000 and (Phi - hist[-1]).y.max() < 1e-4:
        break
    hist.insert(0, Phi)
    hist = hist[:1000]
plt.figure()
plt.plot(Phi.x, Phi.y)
plt.figure()
plt.plot(u.x, u.y)
plt.plot(v.x, v.y)
plt.show()
