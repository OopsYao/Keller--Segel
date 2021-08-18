import fenics as fen
import numpy as np


class ComposeExpression(fen.UserExpression):
    def __init__(self, outer, inner, *args, **kwargs):
        self._outer = outer
        self._inner = inner
        fen.UserExpression.__init__(self, *args, **kwargs)

    def eval(self, value, x):
        value[:] = self._outer(self._inner(x))

    def value_shape(self):
        return self._outer.ufl_shape


def func_from_vertices(mesh, y, flatten=False):
    '''Construct function from vertices values

        mesh -- the corresponding mesh
        y -- function values on the vertices of the mesh,
             the order is the same as the mesh.coordinates().
             For vector function, y is (n, d) array, where d
             is number of vectices and d is the dimension.
    '''
    if isinstance(y, np.ndarray) and len(y.shape) > 1:
        V = fen.VectorFunctionSpace(mesh, 'P', 1)
    else:
        V = fen.FunctionSpace(mesh, 'P', 1)
    d2v = fen.dof_to_vertex_map(V)
    u = fen.Function(V)
    y = np.array(y)
    u.vector()[:] = y.flatten()[d2v]
    return u


def compose(mesh, outer, inner, d):
    '''Compose inner function and outer function
        mesh -- mesh of inner function
        d -- value shape of inner function'''
    inner_value = inner.compute_vertex_values(mesh)
    unflatten = (d, int(inner_value.shape[0] / d))
    outer.set_allow_extrapolation(True)
    outer_value = np.array([
        outer(p) for p in inner_value.reshape(unflatten).T])

    composed = func_from_vertices(mesh, outer_value)
    return composed
