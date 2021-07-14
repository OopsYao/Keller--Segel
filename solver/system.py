import numpy as np


class System:
    def __init__(self, V, c, m):
        self.V = V
        self.c = c
        self.m = m
        pass

    def get_w(self):
        '''Corresponding w of V'''
        return np.linspace(0, self.m, len(self.V))

    def get_dw(self):
        '''dw of the mesh corresponding to V'''
        return self.m / (len(self.V) - 1)

    def get_rho(self):
        '''Recovered rho by V'''
        dw = self.get_dw()
        return dw / np.diff(self.V)

    def get_x(self):
        '''Corresponding x of rho'''
        return (self.V[1:] + self.V[:-1]) / 2

    def get_c_mesh(self):
        '''Basis mesh of the hat function, which is different from physical mesh
        (i.e., the corresponding x, which is not equidistant)'''
        # TODO: Be caution with the array length
        return np.linspace(self.V[0], self.V[-1], len(self.c))

    def get_rho_h(self):
        '''Step interpolation of rho'''
        pass

    def get_c_h(self):
        '''Representation of c by hat function basis'''
        pass


def as_time(system_list):
    T = len(system_list)
    VV = []
    cc = []
    mm = []
    xx = []
    ww = []
    rr = []
    cm = []

    for i, system in enumerate(system_list):
        VV.append(system.V)
        cc.append(system.c)
        mm.append(system.m)
        xx.append(system.get_x())
        ww.append(system.get_w())
        rr.append(system.get_rho())
        cm.append(system.get_c_mesh())

    return np.array(VV), np.array(cc), np.array(mm), np.array(rr), np.array(xx), np.array(ww), np.array(cm)
