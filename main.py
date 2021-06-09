import numpy as np
import matplotlib.pyplot as plt


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
    TV = TV - chi / np.pi * dw * inv_V_ij.sum(-1)
    DT = DT + chi / np.pi * dw * (inv_V_ij ** 2).sum(-1)

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
    dw = m / M
    tilde_V = implicit(dt, dw, V)
    V = explicit(dt, dw, V, tilde_V)
    return (V, c, m)


def operator_S(dt, V, c, m):
    dw = m / M
    log_V_ij = np.log(np.abs(mask(V, 1)))
    c = -1 / np.pi * dw * log_V_ij.sum(-1)
    return V, c, m


def CFL_condition(dw, V):
    CFL = 0.49
    K = 100
    inv_V_ij = 1 / mask(V, np.inf)
    cV = - 1 / np.pi * dw * inv_V_ij.sum(-1)
    return CFL * min((np.diff(V) / np.abs(np.diff(cV))).min(-1) / chi, K * dw)


def step(V, c, m):
    while(True):
        dt = CFL_condition(m / M, V)
        V, c, m = operator_T(dt / 2, V, c, m)
        V, c, m = operator_S(dt, V, c, m)
        V, c, m = operator_T(dt / 2, V, c, m)
        yield dt, V, c, m


def V2rho(dw, V):
    # Note that len(rho) = len(V) - 1
    return dw / np.diff(V)


D_rho = 1
chi = 2.5 * np.pi
gamma = 1
M = 50

m0 = 1
w0 = np.linspace(0, m0, M)
V0 = (w0 - 0.5) / ((w0 + 0.01) * (1.01 - w0)) ** (1 / 4)
c0 = -1 / np.pi * m0 / M * np.log(np.abs(mask(V0, 1))).sum(-1)

N = 3000
VV = np.empty((1 + N, V0.shape[0]))
VV[0] = V0
cc = np.empty((1 + N, c0.shape[0]))
cc[0] = c0
mm = np.empty(1 + N)
mm[0] = m0
dts = np.empty(N)

for (i, (dt, V, c, m)) in zip(range(N), step(V0, c0, m0)):
    VV[1 + i] = V
    cc[1 + i] = c
    mm[1 + i] = m0
    dts[i] = dt
t = np.insert(np.cumsum(dts), 0, 0)

plt.figure('rho')
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho$')
plt.plot((VV[0, 1:] + VV[0, :-1]) / 2,
         V2rho(mm[0] / M, VV[0]), label=f't={t[0]}')
plt.plot((VV[-1, 1:] + VV[-1, :-1]) / 2,
         V2rho(mm[-1] / M, VV[-1]), label=f't={t[-1]:.2f}')
plt.legend()

plt.figure('m')
plt.xlabel(r'$t$')
plt.ylabel(r'$m$')
plt.plot(t, mm)

plt.figure('V')
plt.xlabel(r'$w$')
plt.ylabel(r'$V$')
plt.plot(np.linspace(0, mm[0], M), VV[0], label=f't=0')
plt.plot(np.linspace(0, mm[-1], M), VV[-1], label=f't={t[-1]:.2f}')
plt.legend()

plt.figure('dt')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta t$')
plt.plot(t[:-1], dts)

plt.show()
