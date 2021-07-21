import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from solver.system import System, as_time
from solver.operators import OperatorFactory
import solver.utils as utils
import solver.quadratic as quadratic


class Context:
    pass


class Expr:
    def __init__(self, context) -> None:
        oper_fact = OperatorFactory(context)
        self.S = oper_fact.operator_S
        self.T = oper_fact.operator_T

        def stepper(system):
            while(True):
                dt = CFL_condition(system.get_dw(), system.V, context, system)
                system = self.T(dt / 2, system)
                system = self.S(dt, system)
                system = self.T(dt / 2, system)
                yield dt, system
        self.stepper = stepper

    def as_iterable(self, system0):
        return self.stepper(system0)

    def as_fixed_length(self, system0, I):
        sys_list = [system0]
        dts = []
        for _, (dt, system) in zip(tqdm(range(I)), self.stepper(system0)):
            sys_list.append(system)
            dts.append(dt)
        t = np.insert(np.array(dts).cumsum(), 0, 0)
        return t, sys_list


def CFL_condition(dw, V, context, system):
    if hasattr(context, 'TDT'):
        return 0.001
    else:
        CFL = 0.49
        K = 100
        inv_V_ij = 1 / utils.mask(V, np.inf)
        if hasattr(context, 'cls') and context.cls == 'plain':
            cV = - 1 / np.pi * dw * inv_V_ij.sum(-1)
        else:
            cV = utils.spline_interpolate(system.get_c_mesh(), system.c, V, 1)
        return CFL * min((np.diff(V) / np.abs(np.diff(cV))).min(-1) / context.chi, K * dw)


def init_system(context):
    m0 = 1
    w0 = np.linspace(0, m0, context.M)
    V0 = (w0 - 0.5) / ((w0 + 0.01) * (1.01 - w0)) ** (1 / 4)
    c0 = -1 / np.pi * m0 / (V0.shape[-1] - 1) * \
        np.log(np.abs(utils.mask(V0, 1))).sum(-1)
    return System(V0, c0, m0)


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


def plot_c(ax, idx_select, cc, cm):
    for i in idx_select:
        plt.plot(cm[i], t[i] * np.ones(cc.shape[-1]), cc[i])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$c$')


def plot_V(ax, idx_select, VV, mm):
    for i in idx_select:
        plt.plot(np.linspace(0, mm[i], VV.shape[-1]),
                 t[i] * np.ones(VV.shape[-1]), VV[i])
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$V$')


if __name__ == '__main__':
    if sys.argv[1] == 'expr1':
        context = Context()
        context.D_rho = 1
        context.chi = 2.5 * np.pi
        context.gamma = 1
        context.M = 50
        context.cls = 'plain'
        context.R_rho = (lambda rho: 0)
        expr = Expr(context)

        I = 1000
        idx_select = np.arange(I)[::I // 10]
        system0 = init_system(context)
        t, sys_list = expr.as_fixed_length(system0, I)
        VV, cc, mm, rr, xx, ww, cm = as_time(sys_list)

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
        context = Context()
        context.D_rho = 1
        context.chi = 2.5 * np.pi
        context.gamma = 1
        context.M = 50
        context.cls = 'plain'
        context.R_rho = (lambda rho: 0.2 * rho * (1 - rho))
        expr = Expr(context)

        idx_select = np.arange(I)[::I // 10]
        system0 = init_system(context)
        t, sys_list = expr.as_fixed_length(system0, I)
        VV, cc, mm, rr, xx, ww, cm = as_time(sys_list)

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
        context = Context()
        context.M = 500
        context.R_rho = (lambda rho: 0)
        context.chi = 2.5 * np.pi
        context.gamma = 2 if sys.argv[-1] == 'expr3' else 1.5
        context.cls = 'plain'
        context.D_rho = 1
        expr = Expr(context)

        I = 800

        system0 = init_system(context)
        t, sys_list = expr.as_fixed_length(system0, I)
        VV, cc, mm, rr, xx, ww, cm = as_time(sys_list)
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
        expr7 = sys.argv[-1] == 'expr7'
        alpha = 0.5 if expr7 else 1
        beta = 1
        M = 45
        N = 45

        m0 = 1
        w0 = np.linspace(0, m0, M)
        V0 = (w0 - 0.5) / ((w0 + 0.01) * (1.01 - w0)) ** (1 / 4)
        if expr7:
            c0 = 1 / (1 + np.exp(-5 * np.linspace(-1.58, 1.58, M)))
        else:
            c0 = 1 - np.exp(-20 * np.linspace(-1.58, 1.58, M) ** 2)
        system0 = System(V0, c0, m0)

        context = Context()
        context.R_c = (lambda rho, c: alpha * rho - beta * c)
        context.R_rho = (lambda rho: 0)
        context.D_rho = 0.1
        context.D_c = 0.01 if expr7 else 0.1
        context.epsilon = 1
        context.gamma = 1
        context.chi = 2.5 if expr7 else 5
        context.cls = 'noc'
        expr = Expr(context)

        I = 400
        t, sys_list = expr.as_fixed_length(system0, I)
        VV, cc, mm, rr, xx, ww, cm = as_time(sys_list)
        idx_select = np.arange(I)[::I // 10]

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)

        plt.figure('c')
        ax = plt.axes(projection='3d')
        plot_c(ax, idx_select, cc, cm)

        plt.figure('dt')
        plt.yscale('log')
        plot_dt(plt.gca(), np.diff(t))
    elif sys.argv[-1] in ['quadratic']:
        system0 = System(quadratic.V0, quadratic.c0, quadratic.m0)
        expr = Expr(quadratic)
        I = 800
        t, sys_list = expr.as_fixed_length(system0, I)
        VV, cc, mm, rr, xx, ww, cm = as_time(sys_list)

        idx_select = np.arange(I)[::I // 10]

        plt.figure('rho')
        ax = plt.axes(projection='3d')
        plot_rho(ax, idx_select, rr, xx)

        plt.figure('V')
        ax = plt.axes(projection='3d')
        plot_V(ax, idx_select, VV, mm)

        plt.figure('c')
        ax = plt.axes(projection='3d')
        plot_c(ax, idx_select, cc, cm)

        plt.figure('dt')
        plt.yscale('log')
        plot_dt(plt.gca(), np.diff(t))
    plt.show()
