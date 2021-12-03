import solver.fd.operators as fd
from solver.fd.spec import DiscreteFunc
import numpy as np
from tqdm import tqdm
import solver.context as ctx
from solver.fd.helper import Animation


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
t = 0
ani_uv = Animation(colors=['red', 'black'], markers=['+', ''],
                   labels=['u', 'v'])
ani_Phi = Animation(markers='+', colors='black')
ani_uv.add(t, u_rec, v)
ani_Phi.add(t, Phi)

with tqdm() as pbar:
    while True:
        Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
        u = fd.post_process(Phi)
        v = fd.implicit_v(v, u.interpolate('next'), dt)
        Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
        t = t + dt
        A, dv = fd.JF_S(v, u.interpolate('next'))
        _, dPhi = fd.JF_T(Phi, v.interpolate('spline'))
        r1, r2 = np.abs(A @ v.y + dv).max(), np.abs(dPhi).max()
        ani_uv.add(f't={t:.2f}', u, v)
        ani_Phi.add(f't={t:.2f}', Phi)

        # Equilibrium
        if r1 < 1e-9 and r2 < 1e-9:
            break
        pbar.set_description(f'Steady: {r1:.2e}, {r2:.2e}')
        pbar.update(1)

ani_uv.save('uv.mp4', 'Saving uv')
ani_Phi.save('Phi.mp4', 'Saving Phi')
