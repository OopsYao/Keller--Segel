import fenics as fen
import numpy as np
import ufl


def post_process(Phi, epsilon=1e-4, degree=4):
    '''Recovers rho from Phi'''
    mesh = Phi.function_space().mesh()
    V = fen.FunctionSpace(mesh, 'P', degree)  # Space of rho composing Phi

    # rho composing Phi
    if epsilon == 0:
        f = 1 / ufl.det(fen.grad(Phi))
        rho_Phi = fen.project(f, V)
    else:
        u = fen.TrialFunction(V)
        v = fen.TestFunction(V)
        a = u * v * fen.dx \
            - epsilon * fen.dot(fen.grad(u), fen.grad(v)) * fen.dx
        f = 1 / ufl.det(fen.grad(Phi)) * v * fen.dx
        rho_Phi = fen.Function(V)
        fen.solve(a == f, rho_Phi)

    # Copy function rho_Phi (to rho) and its space
    mesh_mirror = fen.Mesh(mesh)  # This copys mesh coordinates
    V_mirror = fen.FunctionSpace(mesh_mirror, 'P', degree)
    rho = fen.Function(V_mirror)
    rho.vector()[:] = np.array(rho_Phi.vector())

    # Now we apply the mesh transform to recover rho from rho composing Phi.
    # The domain mesh of rho is not equidistant any more.
    M, d = mesh_mirror.coordinates().shape
    trans_coord = np.array(Phi.compute_vertex_values()).reshape((d, M)).T
    mesh_mirror.coordinates()[:] = trans_coord
    return rho
