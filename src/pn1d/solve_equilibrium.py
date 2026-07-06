from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from .constants import DeviceParameters, SiliconParameters, SolverParameters
from .doping import build_doping_functions
from .mesh import create_interval_mesh, create_scalar_space, ensure_output_dir, locate_contact_facets
from .physics import (
    ContactState,
    biased_contacts,
    compute_observables,
    electron_density_ufl,
    equilibrium_contacts,
    hole_density_ufl,
    recombination_ufl,
)
from .plotting import plot_doping_profile, plot_equilibrium_summary, plot_mixed_zero_bias, write_json


@dataclass
class EquilibriumResult:
    mesh: object
    scalar_space: object
    net_doping: fem.Function
    donor_doping: fem.Function
    acceptor_doping: fem.Function
    equilibrium_potential: fem.Function
    mixed_state: fem.Function
    contacts: ContactState
    observables_poisson: dict[str, np.ndarray]
    observables_mixed: dict[str, np.ndarray]
    output_dir: Path


def _nonlinear_petsc_options(
    solver_params: SolverParameters,
) -> dict[str, object]:
    return {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": solver_params.newton_rtol,
        "snes_atol": solver_params.newton_atol,
        "snes_max_it": solver_params.newton_max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_error_if_not_converged": True,
        "ksp_error_if_not_converged": True,
    }


def _sorted_scalar_field(function: fem.Function) -> tuple[np.ndarray, np.ndarray]:
    coords = function.function_space.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    return coords[order], function.x.array.real[order]


def _constant_function(space: fem.FunctionSpace, value: float, name: str) -> fem.Function:
    field = fem.Function(space, name=name)
    field.x.array[:] = value
    field.x.scatter_forward()
    return field


def _mixed_bc_dofs(W, collapsed_space, facets: np.ndarray, subspace_index: int):
    return fem.locate_dofs_topological((W.sub(subspace_index), collapsed_space), W.mesh.topology.dim - 1, facets)


def build_mixed_boundary_conditions(
    W,
    contacts: ContactState,
    left_facets: np.ndarray,
    right_facets: np.ndarray,
) -> list[fem.DirichletBC]:
    bcs: list[fem.DirichletBC] = []
    for subspace_index, left_value, right_value, label in (
        (0, contacts.phi_left_V, contacts.phi_right_V, "phi"),
        (1, contacts.Fn_left_J, contacts.Fn_right_J, "Fn"),
        (2, contacts.Fp_left_J, contacts.Fp_right_J, "Fp"),
    ):
        collapsed_space, _ = W.sub(subspace_index).collapse()
        left_field = _constant_function(collapsed_space, left_value, f"{label}_left")
        right_field = _constant_function(collapsed_space, right_value, f"{label}_right")
        left_dofs = _mixed_bc_dofs(W, collapsed_space, left_facets, subspace_index)
        right_dofs = _mixed_bc_dofs(W, collapsed_space, right_facets, subspace_index)
        bcs.append(fem.dirichletbc(left_field, left_dofs, W.sub(subspace_index)))
        bcs.append(fem.dirichletbc(right_field, right_dofs, W.sub(subspace_index)))
    return bcs


def solve_equilibrium_poisson(
    domain,
    scalar_space: fem.FunctionSpace,
    net_doping: fem.Function,
    contacts: ContactState,
    params: SiliconParameters,
    solver_params: SolverParameters,
) -> fem.Function:
    fdim = domain.topology.dim - 1
    left_facets, right_facets = locate_contact_facets(domain)
    left_dofs = fem.locate_dofs_topological(scalar_space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(scalar_space, fdim, right_facets)

    phi = fem.Function(scalar_space, name="equilibrium_potential")
    phi.interpolate(
        lambda x: contacts.phi_left_V + (contacts.phi_right_V - contacts.phi_left_V) * x[0] / domain.geometry.x[:, 0].max()
    )

    v = ufl.TestFunction(scalar_space)
    F0 = PETSc.ScalarType(params.F0)
    n = electron_density_ufl(phi, F0, params, solver_params)
    p = hole_density_ufl(phi, F0, params, solver_params)
    residual = (
        params.eps_si * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
        - params.q * (p - n + net_doping) * v * ufl.dx
    )

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(contacts.phi_left_V), left_dofs, scalar_space),
        fem.dirichletbc(PETSc.ScalarType(contacts.phi_right_V), right_dofs, scalar_space),
    ]
    problem = NonlinearProblem(
        residual,
        phi,
        petsc_options_prefix="equilibrium_poisson_",
        bcs=bcs,
        petsc_options=_nonlinear_petsc_options(solver_params),
    )
    problem.solve()
    phi.x.scatter_forward()
    return phi


def solve_mixed_state(
    domain,
    net_doping: fem.Function,
    contacts: ContactState,
    equilibrium_phi: fem.Function,
    params: SiliconParameters,
    solver_params: SolverParameters,
    gamma_m3_per_s: float,
    initial_state: fem.Function | None = None,
) -> fem.Function:
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    W = fem.functionspace(domain, basix.ufl.mixed_element([P1, P1, P1]))
    state = fem.Function(W, name="mixed_state")

    Vphi, map_phi = W.sub(0).collapse()
    VFn, map_Fn = W.sub(1).collapse()
    VFp, map_Fp = W.sub(2).collapse()

    if initial_state is not None:
        state.x.array[:] = initial_state.x.array
    else:
        state.x.array[map_phi] = equilibrium_phi.x.array
        state.x.array[map_Fn] = params.F0
        state.x.array[map_Fp] = params.F0
    state.x.scatter_forward()

    phi, Fn, Fp = ufl.split(state)
    vphi, vn, vp = ufl.TestFunctions(W)

    n = electron_density_ufl(phi, Fn, params, solver_params)
    p = hole_density_ufl(phi, Fp, params, solver_params)
    U = recombination_ufl(phi, Fn, Fp, params, solver_params, gamma_m3_per_s)

    poisson_form = (
        params.eps_si * ufl.dot(ufl.grad(phi), ufl.grad(vphi))
        - params.q * (p - n + net_doping) * vphi
    ) * ufl.dx
    electron_form = (
        params.mu_n * n * ufl.dot(ufl.grad(Fn), ufl.grad(vn))
        + params.q * U * vn
    ) * ufl.dx
    hole_form = (
        params.mu_p * p * ufl.dot(ufl.grad(Fp), ufl.grad(vp))
        - params.q * U * vp
    ) * ufl.dx
    residual = poisson_form + electron_form + hole_form

    left_facets, right_facets = locate_contact_facets(domain)
    bcs = build_mixed_boundary_conditions(W, contacts, left_facets, right_facets)
    problem = NonlinearProblem(
        residual,
        state,
        petsc_options_prefix="mixed_state_",
        bcs=bcs,
        petsc_options=_nonlinear_petsc_options(solver_params),
    )
    problem.solve()
    state.x.scatter_forward()
    return state


def split_mixed_state(state: fem.Function) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = state.sub(0).collapse()
    Fn = state.sub(1).collapse()
    Fp = state.sub(2).collapse()
    x_m, phi_values = _sorted_scalar_field(phi)
    _, Fn_values = _sorted_scalar_field(Fn)
    _, Fp_values = _sorted_scalar_field(Fp)
    return x_m, phi_values, Fn_values, Fp_values


def run_equilibrium(
    silicon: SiliconParameters | None = None,
    device: DeviceParameters | None = None,
    solver_params: SolverParameters | None = None,
) -> EquilibriumResult:
    silicon = silicon or SiliconParameters()
    device = device or DeviceParameters()
    solver_params = solver_params or SolverParameters()

    domain = create_interval_mesh(device)
    scalar_space = create_scalar_space(domain)
    doping = build_doping_functions(domain, scalar_space, device)
    contacts = equilibrium_contacts(silicon, device)

    equilibrium_phi = solve_equilibrium_poisson(
        domain=domain,
        scalar_space=scalar_space,
        net_doping=doping["net"],
        contacts=contacts,
        params=silicon,
        solver_params=solver_params,
    )
    x_m, phi_values = _sorted_scalar_field(equilibrium_phi)
    _, donor_values = _sorted_scalar_field(doping["donor"])
    _, acceptor_values = _sorted_scalar_field(doping["acceptor"])
    _, net_values = _sorted_scalar_field(doping["net"])

    observables_poisson = compute_observables(
        x_m=x_m,
        phi_V=phi_values,
        Fn_J=np.full_like(phi_values, silicon.F0),
        Fp_J=np.full_like(phi_values, silicon.F0),
        params=silicon,
        solver=solver_params,
    )

    mixed_state = solve_mixed_state(
        domain=domain,
        net_doping=doping["net"],
        contacts=contacts,
        equilibrium_phi=equilibrium_phi,
        params=silicon,
        solver_params=solver_params,
        gamma_m3_per_s=0.0,
    )
    x_m_mixed, phi_mixed, Fn_mixed, Fp_mixed = split_mixed_state(mixed_state)
    observables_mixed = compute_observables(
        x_m=x_m_mixed,
        phi_V=phi_mixed,
        Fn_J=Fn_mixed,
        Fp_J=Fp_mixed,
        params=silicon,
        solver=solver_params,
    )

    output_dir = ensure_output_dir(solver_params, "equilibrium")
    plot_doping_profile(
        x_m=x_m,
        donor_m3=donor_values,
        acceptor_m3=acceptor_values,
        net_m3=net_values,
        destination=output_dir / "doping_profile.png",
    )
    plot_equilibrium_summary(
        observables=observables_poisson,
        donor_m3=donor_values,
        acceptor_m3=acceptor_values,
        net_m3=net_values,
        destination=output_dir / "equilibrium_summary.png",
    )
    plot_mixed_zero_bias(
        observables=observables_mixed,
        destination=output_dir / "mixed_zero_bias.png",
    )

    write_json(
        output_dir / "equilibrium_summary.json",
        {
            **asdict(silicon),
            **asdict(device),
            **{**asdict(solver_params), "output_dir": str(solver_params.output_dir)},
            "phi_left_V": contacts.phi_left_V,
            "phi_right_V": contacts.phi_right_V,
            "Fn_range_J": [float(np.min(Fn_mixed)), float(np.max(Fn_mixed))],
            "Fp_range_J": [float(np.min(Fp_mixed)), float(np.max(Fp_mixed))],
        },
    )

    return EquilibriumResult(
        mesh=domain,
        scalar_space=scalar_space,
        net_doping=doping["net"],
        donor_doping=doping["donor"],
        acceptor_doping=doping["acceptor"],
        equilibrium_potential=equilibrium_phi,
        mixed_state=mixed_state,
        contacts=contacts,
        observables_poisson=observables_poisson,
        observables_mixed=observables_mixed,
        output_dir=output_dir,
    )


def main() -> None:
    result = run_equilibrium()
    if MPI.COMM_WORLD.rank == 0:
        print(f"Equilibrium outputs written to {result.output_dir}")


if __name__ == "__main__":
    main()
