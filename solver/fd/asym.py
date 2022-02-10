import numpy as np
from scipy.optimize import root_scalar


def asym(M1, M2, L):
    u_bar = M1 / L
    v_bar = M2 / L

    def chi_k(k):
        return 1 + (k * np.pi / L) ** 2

    def at_chi_k(k, epsilon):
        chi = chi_k(k)
        assert -u_bar / chi <= epsilon <= u_bar / chi

        def u(x):
            return u_bar + epsilon * chi * np.cos(k * np.pi * x / L)

        def v(x):
            return v_bar + epsilon * np.cos(k * np.pi * x / L)
        return u, v

    def otherwise(chi, mirror=False):
        '''Return (numpy) function u and v with given chi (> chi1).'''
        chi1 = chi_k(1)
        assert chi > chi1

        o = np.sqrt(chi - 1)

        # Find l_star
        def eq(el):
            return 1 / o * np.tan(o * el) - np.tanh(el - L)

        lb = np.pi / 2 / o
        ub = np.pi / o
        # eq(lb) = -inf, so move a little.
        # Bigger chi causes smaller adjustment.
        lb_m = lb + (ub - lb) / max(chi, 5)
        l_star = root_scalar(eq, bracket=[lb_m, ub], method='bisect', x0=lb_m).root
        # Check l_star
        assert lb < l_star < ub

        # Construct A, B
        A = u_bar * L / (1 / o * np.sin(o * l_star) - l_star * np.cos(o * l_star))
        B = u_bar * L * (1 / chi - 1) \
            / ((1 / o * np.tan(o * l_star) - l_star) * np.cosh(l_star - L))

        def u(x):
            if mirror:
                x = L - x
            left = A * (np.cos(o * x) - np.cos(o * l_star))
            right = 0
            return np.where(x < l_star, left, right)

        def v(x):
            if mirror:
                x = L - x
            left = A * (np.cos(o * x) / chi - np.cos(o * l_star))
            right = B * np.cosh(x - L)
            return np.where(x < l_star, left, right)
        return u, v
    return at_chi_k, otherwise
