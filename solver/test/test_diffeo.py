import unittest
from solver.fen.diffeo import post_process, pre_process
import fenics as fen


# This test sample is not good, as rho is not continous,
# leading Phi not a function.
def rho_Phi():
    mesh = fen.IntervalMesh(1000, -1, 1)
    V0 = fen.FunctionSpace(mesh, 'P', 2)  # Function space of rho
    M = 1  # Total mass
    rho = fen.Expression(
        'x[0] < -0.5 || x[0] > 0.5 ? 0 : M',
        degree=3, M=M)
    rho = fen.interpolate(rho, V0)

    mesh_tilde = fen.IntervalMesh(1000, 0, M)
    V = fen.VectorFunctionSpace(mesh_tilde, 'P', 1)  # Space of Phi
    Phi = fen.Expression(
        ('x[0] < eps ? -1 :(M - x[0] < eps ? 1 : x[0] / M - 0.5)',),
        degree=1, M=M, eps=1e-11)
    Phi = fen.interpolate(Phi, V)
    return rho, Phi


def rho_Phi2(pse_inv=True):
    M = 1
    mesh = fen.IntervalMesh(1000, -1, 1)
    V0 = fen.FunctionSpace(mesh, 'P', 2)  # Function space of rho
    rho = fen.Expression(
        'x[0] < 0 ? 0 : 2 * M * x[0]', degree=1, M=M if pse_inv else 2)
    rho = fen.interpolate(rho, V0)

    if pse_inv:
        Phi = fen.Expression((
            'x[0] < eps ? -1 : (M - x[0] < eps ? 1 : pow(x[0] / M, 0.5))',),
            degree=3, M=M, eps=1e-11)
        mesh_tilde = fen.IntervalMesh(1000, 0, M)
    else:
        Phi = fen.Expression((
            'x[0] < -1 + eps ? -1 :'
            '(1 - x[0] < eps ? 1 : pow((1 + x[0]) / 2, 0.5))',),
            degree=3, eps=1e-11)
        mesh_tilde = fen.IntervalMesh(1000, -1, 1)
    V = fen.VectorFunctionSpace(mesh_tilde, 'P', 1)  # Space of Phi
    Phi = fen.interpolate(Phi, V)
    return rho, Phi


class TestDiffeo(unittest.TestCase):
    def test_post_process(self):
        rho, Phi = rho_Phi2()
        rho_recover = post_process(Phi, degree=4, epsilon=1e-9)

        rho_recover_proj = fen.project(rho_recover, rho.function_space())
        rho_recover_intr = fen.interpolate(rho_recover, rho.function_space())

        err_proj = fen.errornorm(rho, rho_recover_proj, 'l2')
        err_intr = fen.errornorm(rho, rho_recover_intr, 'l2')
        self.assertAlmostEqual(err_proj, 0, 1)
        self.assertAlmostEqual(err_intr, 0, 1)

    def test_pre_process(self):
        rho, Phi = rho_Phi2(pse_inv=False)
        Phi_recover = pre_process(rho)

        Phi_recover_proj = fen.project(Phi_recover, Phi.function_space())
        Phi_recover_intr = fen.interpolate(Phi_recover, Phi.function_space())
        err_proj = fen.errornorm(Phi, Phi_recover_proj, 'l2')
        err_intr = fen.errornorm(Phi, Phi_recover_intr, 'l2')
        self.assertAlmostEqual(err_proj, 0, 1)
        self.assertAlmostEqual(err_intr, 0, 1)
