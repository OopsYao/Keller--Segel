import solver.fd.operators as fd
from solver.fd.spec import DiscreteFunc
import numpy as np
from tqdm import tqdm
import solver.context as ctx
from solver.fd.helper import Animation, Reducer


def u0(x):
    return 3 / 4 * np.where(1 > (x - np.pi / 2) ** 2,
                            1 - (x - np.pi / 2) ** 2, 0)


def v0(x):
    return 1.2 * np.exp(-3 * x ** 2) \
        + 1.0 * np.exp(-3 * (x - np.pi) ** 2)


a = ctx.a
b = ctx.b
x = np.linspace(a, b, 100)
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


def looks_different(f, g):
    return ((f - g).norm() / f.norm()) > 1e-3


def Phi_different(Phi0, Phi1):
    _, Phi0 = Phi0
    _, Phi1 = Phi1
    return looks_different(Phi0, Phi1)


def uv_different(uv0, uv1):
    _, u0, v0 = uv0
    _, u1, v1 = uv1
    return looks_different(u0, u1) or looks_different(v0, v1)


reducer_Phi = Reducer(Phi_different, 100)
reducer_uv = Reducer(uv_different, 100)
try:
    with tqdm() as pbar:
        while True:
            # Monotonicity preserving check
            assert (Phi.y[1:] >= Phi.y[:-1]).all()
            dt = fd.CFL(Phi, v)
            Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
            u = fd.post_process(Phi)
            v = fd.implicit_v(v, u.interpolate('next'), dt)
            Phi = fd.implicit_Phi(Phi, v.interpolate('spline'), dt / 2)
            t = t + dt
            A, dv = fd.JF_S(v, u.interpolate('next'))
            _, dPhi = fd.JF_T(Phi, v.interpolate('spline'))
            r1, r2 = np.abs(A @ v.y + dv).max(), np.abs(dPhi).max()

            reducer_Phi.add((t, Phi))
            reducer_uv.add((t, u, v))

            # Equilibrium
            if r1 < 1e-9 and r2 < 2.2e-8:
                break
            pbar.set_description(f'Steady: {r1:.2e}, {r2:.2e}')
            pbar.update(1)
except KeyboardInterrupt:
    pass

for t, Phi in reducer_Phi.retrieve():
    ani_Phi.add(f't={t:.2f}', Phi)
for t, u, v in reducer_uv.retrieve():
    ani_uv.add(f't={t:.2f}', u, v)


ani_uv.save('uv.mp4', 'Saving uv')
ani_Phi.save('Phi.mp4', 'Saving Phi')
