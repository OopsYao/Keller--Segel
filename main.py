import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from tqdm import tqdm


def mask(V, mask_value):
    V_ij = np.expand_dims(V, -1) - V
    mask = np.abs(V_ij) >= 1e-6
    tran_V_ij = np.where(mask, V_ij, mask_value)
    return tran_V_ij


def TDT(dw, V):
    dV = 1 / np.diff(V)
    dV_p = np.append(dV, 0)
    dV_m = np.insert(dV, 0, 0)

    main_diag = dV_p ** (gamma + 1) + dV_m ** (gamma + 1)
    dV_gamma = dV ** (gamma + 1)
    TV = -D_rho / gamma * \
        dw ** (gamma - 1) * (dV_p ** gamma - dV_m ** gamma)
    DT = D_rho * dw ** (gamma - 1) * \
        (np.diag(dV_gamma, -1) - np.diag(main_diag) + np.diag(dV_gamma, 1))

    inv_V_ij = 1 / mask(V, np.inf)
    inv_V_ij2 = inv_V_ij ** 2
    TV = TV - chi / np.pi * dw * inv_V_ij.sum(-1)
    DT = DT + chi / np.pi * dw * (np.diag(inv_V_ij2.sum(-1)) - inv_V_ij2)

    return TV, DT


def implicit(dt, dw, V):
    tilde_V = V
    while(True):
        TV, DT = TDT(dw, tilde_V)
        TV = TV - 2 * (tilde_V - V) / dt
        DT = DT - 2 / dt * np.identity(len(V))

        delta = np.linalg.solve(DT, -TV)
        tilde_V = tilde_V + delta
        if np.abs(delta).max() <= 1e-7:
            break
    return tilde_V


def explicit(dt, dw, V, tilde_V):
    TV, DT = TDT(dw, tilde_V)
    return V + dt * TV


def operator_T(dt, V, c, m):
    dw = m / (V.shape[-1] - 1)
    tilde_V = implicit(dt, dw, V)
    V = explicit(dt, dw, V, tilde_V)
    return (V, c, m)


def operator_S(dt, V, c, m):
    def R_rho(rho):
        # mu comes from outside
        return mu * rho * (1 - rho)
    dw = m / (V.shape[-1] - 1)

    # Step 1
    rho = V2rho(dw, V)
    c = -1 / np.pi * dw * np.log(np.abs(mask(V, 1))).sum(-1)
    tilde_rho = rho + dt / 2 * R_rho(rho)

    # Step 2
    rho = rho + dt * R_rho(tilde_rho)
    m = (rho * np.diff(V)).sum(-1)

    V = rho2V(rho, V)
    return V, c, m


def CFL_condition(dw, V):
    CFL = 0.49
    K = 100
    inv_V_ij = 1 / mask(V, np.inf)
    cV = - 1 / np.pi * dw * inv_V_ij.sum(-1)
    return CFL * min((np.diff(V) / np.abs(np.diff(cV))).min(-1) / chi, K * dw)


def step(V, c, m):
    while(True):
        dt = CFL_condition(m / (V.shape[-1] - 1), V)
        V, c, m = operator_T(dt / 2, V, c, m)
        V, c, m = operator_S(dt, V, c, m)
        V, c, m = operator_T(dt / 2, V, c, m)
        yield dt, V, c, m


def V2rho(dw, V):
    # Note that len(rho) = len(V) - 1
    dw = np.expand_dims(dw, -1)
    return dw / np.diff(V)


def rho2V(rho, V):
    # V is the x-points of interpolation
    bins = np.insert((rho * np.diff(V)).cumsum(-1), 0, 0)
    m = (rho * np.diff(V)).sum(-1)
    w = np.linspace(0, m, bins.shape[-1])
    inds = np.minimum(np.digitize(w, bins), bins.shape[-1] - 1)
    V = ((w - bins[inds -  1])  * V[inds] + (bins[inds]  - w) * V[inds - 1]) / (bins[inds] - bins[inds - 1])
    return V


def full(N, V0, c0, m0):
    VV = np.empty((1 + N, V0.shape[-1]))
    VV[0] = V0
    cc = np.empty((1 + N, c0.shape[-1]))
    cc[0] = c0
    mm = np.empty(1 + N)
    mm[0] = m0
    dts = np.empty(N)
    for (i, (dt, V, c, m)) in zip(tqdm(range(N)), step(V0, c0, m0)):
        VV[1 + i] = V
        cc[1 + i] = c
        mm[1 + i] = m
        dts[i] = dt
    return dts, VV, cc, mm


def init_system():
    m0 = 1
    w0 = np.linspace(0, m0, M)
    V0 = (w0 - 0.5) / ((w0 + 0.01) * (1.01 - w0)) ** (1 / 4)
    c0 = -1 / np.pi * m0 / (V0.shape[-1] - 1) * \
        np.log(np.abs(mask(V0, 1))).sum(-1)
    return V0, c0, m0


def plot_m(ax, t, mm):
    ax.plot(t, mm)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$m$')


def plot_dt(ax, dts):
    ax.plot(np.insert(np.cumsum(dts), 0, 0)[:-1], dts)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Delta t$')


def plot_rho(ax, idx_select, rr, xx):
    for i in idx_select:
        ax.plot(xx[i], t[i] * np.ones_like(xx[i]), rr[i])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$\rho$')


def plot_V(ax, idx_select, VV, mm):
    for i in idx_select:
        plt.plot(np.linspace(0, mm[i], VV.shape[-1]),
                 t[i] * np.ones(VV.shape[-1]), VV[i])
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$V$')


D_rho = 1
chi = 2.5 * np.pi
gamma = 1
M = 50
V0, c0, m0 = init_system()

if sys.argv[1] == 'expr1':
    N = 1000
    mu = 0
    dts, VV, cc, mm = full(N, V0, c0, m0)
    t = np.insert(np.cumsum(dts), 0, 0)
    idx_select = np.arange(N)[::N // 4]

    plt.figure('rho')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$')
    for i in idx_select:
        plt.plot((VV[i, 1:] + VV[i, :-1]) / 2,
                 V2rho(mm[i] / (VV.shape[-1] - 1), VV[i]), label=f't={t[i]:.2f}')
    plt.legend()

    plt.figure('V')
    plt.xlabel(r'$w$')
    plt.ylabel(r'$V$')
    for i in idx_select:
        plt.plot(np.linspace(0, mm[i], VV.shape[-1]),
                 VV[i], label=f't={t[i]:.2f}')
    plt.legend()

    plt.figure('dt')
    plt.yscale('log')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Delta t$')
    plt.plot(t[:-1], dts)

elif sys.argv[1] == 'expr2':
    N = 5000
    mu = 0.2
    dts, VV, cc, mm = full(N, V0, c0, m0)
    rr = V2rho(mm / (VV.shape[-1] - 1), VV)  # rho with time t (1st axis)
    xx = (VV[:, 1:] + VV[:, :-1]) / 2  # x with time t (1st axis)
    t = np.insert(np.cumsum(dts), 0, 0)
    idx_select = np.arange(N)[::N // 10]

    plt.figure('rho')
    ax = plt.axes(projection='3d')
    plot_rho(ax, idx_select, rr, xx)

    plt.figure('V')
    ax = plt.axes(projection='3d')
    plot_V(ax, idx_select, VV, mm)

    plt.figure('m')
    plot_m(plt.gca(), t, mm)

    plt.figure('dt')
    plot_dt(plt.gca(), dts)

elif sys.argv[-1] in ['expr3', 'expr4']:
    M = 500
    N = 800 if sys.argv[-1] == 'expr3' else 700
    mu = 0
    gamma = 2 if sys.argv[-1] == 'expr3' else 1.5
    V0, c0, m0 = init_system()
    dts, VV, cc, mm = full(N, V0, c0, m0)
    rr = V2rho(mm / (VV.shape[-1] - 1), VV)  # rho with time t (1st axis)
    xx = (VV[:, 1:] + VV[:, :-1]) / 2  # x with time t (1st axis)
    t = np.insert(np.cumsum(dts), 0, 0)
    idx_select = np.arange(N)[::N // 10]

    plt.figure('rho')
    ax = plt.axes(projection='3d')
    plot_rho(ax, idx_select, rr, xx)

    plt.figure('V')
    ax = plt.axes(projection='3d')
    plot_V(ax, idx_select, VV, mm)

    plt.figure('dt')
    plt.yscale('log')
    plot_dt(plt.gca(), dts)
plt.show()
