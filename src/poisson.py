"""Poisson solve with DOLFINx and PyVista visualization."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

try:
    import dolfinx
    import pyvista
    import ufl
    from dolfinx import fem, mesh, plot
except ImportError as exc:  # pragma: no cover - documents optional dependency
    raise SystemExit(
        "Install dolfinx and pyvista in a compatible environment before running."
    ) from exc


def main() -> None:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
    v_space = fem.functionspace(domain, ("Lagrange", 1))

    facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(v_space, domain.topology.dim - 1, facets)
    bc = fem.dirichletbc(fem.Constant(domain, 0.0), dofs, v_space)

    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    x = ufl.SpatialCoordinate(domain)
    f = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l_form = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, l_form, bcs=[bc])
    uh = problem.solve()

    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    topology, cell_types, geometry = plot.vtk_mesh(v_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["u"] = uh.x.array.real
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, scalars="u", show_edges=False)
    plotter.view_xy()
    plotter.screenshot(results / "poisson_solution.png")


if __name__ == "__main__":
    main()
