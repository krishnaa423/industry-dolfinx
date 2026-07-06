from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from .constants import DeviceParameters, SolverParameters


def create_interval_mesh(device: DeviceParameters) -> mesh.Mesh:
    return mesh.create_interval(MPI.COMM_WORLD, device.num_cells, np.array([0.0, device.length_m], dtype=np.float64))


def create_scalar_space(domain: mesh.Mesh, degree: int = 1) -> fem.FunctionSpace:
    return fem.functionspace(domain, ("Lagrange", degree))


def locate_contact_facets(domain: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    fdim = domain.topology.dim - 1
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    right = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], domain.geometry.x[:, 0].max())
    )
    return left, right


def ensure_output_dir(solver: SolverParameters, name: str) -> Path:
    path = solver.output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path
