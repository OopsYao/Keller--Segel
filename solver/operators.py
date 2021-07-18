import numpy as np
from solver.system import System
import solver.utils as utils


class OperatorFactory:

    def __init__(self, context) -> None:
        self.context = context

    def __TDT_diffusion(self, dw, V):
        gamma = self.context.gamma
        D_rho = self.context.D_rho
        dV = 1 / np.diff(V)
        TV = -D_rho / gamma * utils.V_rooster(dV, dw, gamma)
        DT = D_rho * utils.matrix_elephant(dV, dw, gamma + 1)
        return TV, DT

    def __TDT_given_c(self, dw, V):
        chi = self.context.chi
        inv_V_ij = 1 / utils.mask(V, np.inf)
        inv_V_ij2 = inv_V_ij ** 2
        TV = - chi / np.pi * dw * inv_V_ij.sum(-1)
        DT = chi / np.pi * dw * (np.diag(inv_V_ij2.sum(-1)) - inv_V_ij2)
        return TV, DT

    def __TDT_external_c(self, V, c, c_mesh):
        chi = self.context.chi
        TV = chi * utils.spline_interpolate(c_mesh, c, V, 1)
        DT = np.diag(chi * utils.spline_interpolate(c_mesh, c, V, 2))
        return TV, DT

    def __over_TDT_given_c(self, dw, V):
        TV_diff, DT_diff = self.__TDT_diffusion(dw, V)
        TV_c, DT_c = self.__TDT_given_c(dw, V)
        return TV_diff + TV_c, DT_diff + DT_c

    def __over_TDT_external_c(self, dw, V, c, c_mesh):
        TV_diff, DT_diff = self.__TDT_diffusion(dw, V)
        TV_c, DT_c = self.__TDT_external_c(V, c, c_mesh)
        return TV_diff + TV_c, DT_diff + DT_c

    def operator_T(self, dt, system):
        dw = system.get_dw()

        # TODO: Move this judgement outside
        cls = self.context.cls
        if cls == 'plain':
            def TDT(V):
                return self.__over_TDT_given_c(dw, V)
        else:
            def TDT(V):
                return self.__over_TDT_external_c(dw, V, system.c, system.get_c_mesh())

        # implicit
        tilde_V = system.V
        while(True):
            TV, DT = TDT(tilde_V)
            TV = TV - 2 * (tilde_V - system.V) / dt
            DT = DT - 2 / dt * np.identity(len(system.V))

            delta = np.linalg.solve(DT, -TV)
            tilde_V = tilde_V + delta
            if np.abs(delta).max() <= 1e-7:
                break

        # explicit
        TV, DT = TDT(tilde_V)
        V = system.V + dt * TV

        new_sys = System(V, system.c, system.m)
        return new_sys

    def operator_S(self, dt, system):
        R_rho = self.context.R_rho
        dw = system.get_dw()
        cls = self.context.cls

        # rho and m
        rho = system.get_rho()
        tilde_rho = rho + dt / 2 * R_rho(rho)

        rho = rho + dt * R_rho(tilde_rho)

        # c
        # TODO: Move this judgement outside
        if cls == 'plain':
            c = -1 / np.pi * dw * \
                np.log(np.abs(utils.mask(system.V, 1))).sum(-1)
        else:
            epsilon = self.context.epsilon
            R_c = self.context.R_c
            D_c = self.context.D_c

            x = system.get_c_mesh()
            dx = x[1] - x[0]
            M = utils.hat_inner_product(x)
            L = utils.hat_der_inner_product(x)
            tilde_c = np.linalg.solve(2 * epsilon / dt * M + D_c * L,
                                      dx * R_c(utils.hat_interpolate(system.get_x(), rho, x), system.c))

            c = np.linalg.solve(epsilon / dt * M,
                                epsilon / dt * M @ system.c - D_c * L @ tilde_c +
                                dx * R_c(utils.hat_interpolate(system.get_x(), tilde_rho, x), tilde_c))

        m = (rho * np.diff(system.V)).sum(-1)
        V = self.__rho2V(rho, system.V)
        return System(V, c, m)

    def __rho2V(self, rho, V):
        # V is the x-points of interpolation
        bins = np.insert((rho * np.diff(V)).cumsum(-1), 0, 0)
        m = (rho * np.diff(V)).sum(-1)
        w = np.linspace(0, m, bins.shape[-1])
        inds = np.minimum(np.digitize(w, bins), bins.shape[-1] - 1)
        V = ((w - bins[inds - 1]) * V[inds] + (bins[inds] - w)
             * V[inds - 1]) / (bins[inds] - bins[inds - 1])
        return V
