import unittest
from solver.fen.utils import ComposeExpression, compose, func_from_vertices
import fenics as fen
import mshr as mh
import numpy as np

domain = mh.Circle(fen.Point(0, 0), 1)
mesh = mh.generate_mesh(domain, 30)
x = mesh.coordinates()

V = fen.FunctionSpace(mesh, 'P', 2)
v = fen.Expression('pow(x[0], 2) + 2 * pow(x[1], 2)', degree=2)
v = fen.interpolate(v, V)
VV = fen.VectorFunctionSpace(mesh, 'P', 1)
gv = fen.project(fen.grad(v), VV)
gvgv = fen.interpolate(fen.Expression(('4 * x[0]', '16 * x[1]'), degree=1), VV)
vgv = fen.interpolate(
    fen.Expression('4 * pow(x[0], 2) + 32 * pow(x[1], 2)', degree=1), V)


class TestFenUtils(unittest.TestCase):
    def test_scalar_func_from_vertices(self):
        y = x[:, 0] ** 2 + 2 * x[:, 1] ** 2
        u = func_from_vertices(mesh, y)

        err = fen.errornorm(u, v, 'L2')
        self.assertAlmostEqual(0, err, places=2)

    def test_vector_func_from_vertices(self):
        gu = func_from_vertices(mesh,
                                np.stack((2 * x[:, 0], 4 * x[:, 1]), axis=-1))

        # TODO vector func norm ??
        err = fen.errornorm(gu, gv, 'L2')
        self.assertAlmostEqual(0, err)

    def test_scalar_compose(self):
        ugu = compose(mesh, v, gv, 2)
        err = fen.errornorm(ugu, vgv, 'L2')
        self.assertAlmostEqual(0, err, places=1)

    def test_vector_compose(self):
        gugu = compose(mesh, gv, gv, 2)
        err = fen.errornorm(gugu, gvgv, 'L2')
        self.assertAlmostEqual(0, err)

    def test_vector_ComposeExpression(self):
        gugu = ComposeExpression(gv, gv)
        gv.set_allow_extrapolation(True)
        gugu = fen.project(gugu, VV)

        err = fen.errornorm(gugu, gvgv, 'L2')
        self.assertAlmostEqual(0, err)

    def test_scalar_ComposeExpression(self):
        ugu = ComposeExpression(v, gv)
        v.set_allow_extrapolation(True)
        ugu = fen.project(ugu, V)

        err = fen.errornorm(ugu, vgv, 'L2')
        self.assertAlmostEqual(0, err)
