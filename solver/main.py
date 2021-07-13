import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from tqdm import tqdm
from solver.system import System, as_time
import solver.utils as utils


def mask(V, mask_value):
    V_ij = np.expand_dims(V, -1) - V
    mask = np.abs(V_ij) >= 1e-6
    tran_V_ij = np.where(mask, V_ij, mask_value)
    return tran_V_ij


def TDT(dw, V, c, c_mesh):
    dV = 1 / np.diff(V)
    dV_p = np.append(dV, 0)
    dV_m = np.insert(dV, 0, 0)

    main_diag = dV_p ** (gamma + 1) + dV_m ** (gamma + 1)
    dV_gamma = dV ** (gamma + 1)
    TV = -D_rho / gamma * \
        dw ** (gamma - 1) * (dV_p ** gamma - dV_m ** gamma)
    DT = D_rho * dw ** (gamma - 1) * \
        (np.diag(dV_gamma, -1) - np.diag(main_diag) + np.diag(dV_gamma, 1))

    TV = TV + chi * utils.spline_interpolate(c_mesh, c, V, 1)
    DT = DT + np.diag(chi * utils.spline_interpolate(c_mesh, c, V, 2))

    # inv_V_ij = 1 / mask(V, np.inf)
    # inv_V_ij2 = inv_V_ij ** 2
    # TV = TV - chi / np.pi * dw * inv_V_ij.sum(-1)
    # DT = DT + chi / np.pi * dw * (np.diag(inv_V_ij2.sum(-1)) - inv_V_ij2)

    return TV, DT


def implicit(dt, dw, V, c, c_mesh):
    tilde_V = V
    while(True):
        TV, DT = TDT(dw, V, c, c_mesh)
        TV = TV - 2 * (tilde_V - V) / dt
        DT = DT - 2 / dt * np.identity(len(V))

        delta = np.linalg.solve(DT, -TV)
        tilde_V = tilde_V + delta
        if np.abs(delta).max() <= 1e-7:
            break
    return tilde_V


def explicit(dt, dw, V, tilde_V, c, c_mesh):
    TV, DT = TDT(dw, tilde_V, c, c_mesh)
    return V + dt * TV


def operator_T(dt, system):
    dw = system.get_dw()
    tilde_V = implicit(dt, dw, system.V, system.c, system.get_c_mesh())
    V = explicit(dt, dw, system.V, tilde_V, system.c, system.get_c_mesh())
    new_sys = System(V, system.c, system.m)
    return new_sys


def operator_S(dt, system):
    def R_rho(rho):
        # mu comes from outside
        return mu * rho * (1 - rho)

    def R_c(rho, c):
        alpha = 0.5
        beta = 0.1
        return alpha * rho - beta * c
    epsilon = 1
    dw = system.get_dw()

    # Step 1
    rho = system.get_rho()
    tilde_rho = rho + dt / 2 * R_rho(rho)

    x = system.get_c_mesh()
    M = utils.hat_inner_product(x)
    L = utils.hat_der_inner_product(x)
    tilde_c = np.linalg.solve(2 * epsilon / dt * M + D_c * L,
                              R_c(utils.hat_interpolate(system.get_x(), rho, x), system.c))

    # Step 2
    rho = rho + dt * R_rho(tilde_rho)
    m = (rho * np.diff(system.V)).sum(-1)
    c = np.linalg.solve(epsilon / dt * M,
                        epsilon / dt * M @ system.c - D_c * L @ tilde_c +
                        R_c(utils.hat_interpolate(system.get_x(), tilde_rho, x), tilde_c))

    V = rho2V(rho, system.V)
    return System(V, c, m)


def CFL_condition(dw, V):
    CFL = 0.49
    K = 100
    inv_V_ij = 1 / mask(V, np.inf)
    cV = - 1 / np.pi * dw * inv_V_ij.sum(-1)
    return CFL * min((np.diff(V) / np.abs(np.diff(cV))).min(-1) / chi, K * dw)


def step(system):
    while(True):
        dt = CFL_condition(system.get_dw(), system.V)
        system = operator_T(dt / 2, system)
        system = operator_S(dt, system)
        system = operator_T(dt / 2, system)
        yield dt, system


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
    V = ((w - bins[inds - 1]) * V[inds] + (bins[inds] - w)
         * V[inds - 1]) / (bins[inds] - bins[inds - 1])
    return V


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


def full(I, system0):
    sys_list = [system0]
    dts = []
    for i, (dt, system) in zip(tqdm(range(I)), step(system0)):
        sys_list.append(system)
        dts.append(dt)
    t = np.insert(np.array(dts).cumsum(), 0, 0)
    return t, sys_list


if __name__ == '__main__':
    D_rho = 1
    chi = 2.5 * np.pi
    gamma = 1
    M = 50
    V0, c0, m0 = init_system()
    system0 = System(V0, c0, m0)

    if sys.argv[1] == 'expr1':
        I = 1000
        mu = 0

        idx_select = np.arange(I)[::I // 10]
        system0 = System(V0, c0, m0)
        t, sys_list = full(I, system0)
        VV, cc, mm, rr, xx, ww = as_time(sys_list)

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)

        plt.figure('dt')
        plt.yscale('log')
        plot_dt(plt.gca(), np.diff(t))

    elif sys.argv[1] == 'expr2':
        I = 5000
        mu = 0.2

        idx_select = np.arange(I)[::I // 10]
        t, sys_list = full(I, system0)
        VV, cc, mm, rr, xx, ww = as_time(sys_list)

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)

        plt.figure('m')
        plot_m(plt.gca(), t, mm)

        plt.figure('dt')
        plot_dt(plt.gca(), np.diff(t))

    elif sys.argv[-1] in ['expr3', 'expr4']:
        M = 500
        I = 800
        mu = 0
        gamma = 2 if sys.argv[-1] == 'expr3' else 1.5

        V0, c0, m0 = init_system()
        t, sys_list = full(I, system0)
        VV, cc, mm, rr, xx, ww = as_time(sys_list)
        idx_select = np.arange(I)[::I // 10]

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)

        plt.figure('dt')
        plt.yscale('log')
        plot_dt(plt.gca(), np.diff(t))

    elif sys.argv[-1] in ['expr7', 'expr8']:
        D_rho = 0.1
        D_c = 0.01
        chi = 2.5
        M = 45
        N = 45
        I = 800
        mu = 0
        gamma = 1

        V0, c0, m0 = init_system()
        c0 = 1 / (1 + np.exp(-5 * np.linspace(-1.58, 1.58, N)))
        system0 = System(V0, c0, m0)
        t, sys_list = full(I, system0)
        VV, cc, mm, rr, xx, ww = as_time(sys_list)
        idx_select = np.arange(I)[::I // 10]

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)
        plt.figure('dt')
        plt.yscale('log')
        plot_dt(plt.gca(), np.diff(t))

    plt.show()
