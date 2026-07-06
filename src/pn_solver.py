from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from .device_config import DeviceConfig
from .simulation_data import BiasPlotData, BiasState, ContactValues


class PnJunctionSolver:
    def __init__(self, config: DeviceConfig) -> None:
        self.config = config

    def solve_bias(self, bias_V: float, previous_state: BiasState | None) -> BiasState:
        domain = previous_state.domain if previous_state is not None else self._create_domain()
        donor, acceptor = (
            (previous_state.donor, previous_state.acceptor)
            if previous_state is not None
            else self._create_doping_functions(domain)
        )
        contacts = self._biased_contacts(bias_V)
        solution = self._build_initial_solution(domain, donor, acceptor, contacts, previous_state)

        phi, Fn, Fp = ufl.split(solution)
        vphi, vn, vp = ufl.TestFunctions(solution.function_space)
        n = self._electron_density_ufl(phi, Fn)
        p = self._hole_density_ufl(phi, Fp)
        recombination = self._recombination_ufl(n, p)
        transport_floor = max(self.config.ni_au * 1.0e-6, 1.0e-30)
        n_transport = ufl.max_value(n, transport_floor)
        p_transport = ufl.max_value(p, transport_floor)

        residual = (
            (
                self.config.epsilon_r * ufl.dot(ufl.grad(phi), ufl.grad(vphi))
                - self.config.area_au * (p - n + donor - acceptor) * vphi
            )
            + (
                self.config.mu_n_scale * n_transport * ufl.dot(ufl.grad(Fn), ufl.grad(vn))
                + self.config.area_au * recombination * vn
            )
            + (
                -self.config.mu_p_scale * p_transport * ufl.dot(ufl.grad(Fp), ufl.grad(vp))
                + self.config.area_au * recombination * vp
            )
        ) * ufl.dx

        left_facets, right_facets = self._locate_contact_facets(domain)
        problem = NonlinearProblem(
            residual,
            solution,
            bcs=self._boundary_conditions(solution.function_space, contacts, left_facets, right_facets),
            petsc_options_prefix=f"pn_bias_{self.bias_label(bias_V)}_",
            petsc_options=self._nonlinear_solver_options(),
        )

        converged = True
        try:
            problem.solve()
            solution.x.scatter_forward()
        except PETSc.Error:
            if previous_state is None:
                raise
            converged = False
            solution.x.scatter_forward()

        return BiasState(domain, donor, acceptor, solution, contacts, converged)

    def build_plot_data(self, state: BiasState, bias_V: float) -> BiasPlotData:
        x_au, phi_ha, Fn_ha, Fp_ha = self._extract_solution_components(state)
        donor_au = self._donor_density_profile(x_au)
        acceptor_au = self._acceptor_density_profile(x_au)
        n_au = self._electron_density_numpy(phi_ha, Fn_ha)
        p_au = self._hole_density_numpy(phi_ha, Fp_ha)
        Jn_density_au = self.config.mu_n_scale * n_au * np.gradient(Fn_ha, x_au, edge_order=2)
        Jp_density_au = self.config.mu_p_scale * p_au * np.gradient(Fp_ha, x_au, edge_order=2)
        total_current_A = self._current_to_amp(self.config.area_au * (Jn_density_au + Jp_density_au))
        return BiasPlotData(
            bias_V=bias_V,
            converged=state.converged,
            x_um=self._length_to_um(x_au),
            phi_ha=phi_ha,
            Fn_ha=Fn_ha,
            Fp_ha=Fp_ha,
            donor_m3=self._density_to_m3(donor_au),
            acceptor_m3=self._density_to_m3(acceptor_au),
            total_current_A=total_current_A,
        )

    def bias_label(self, bias_V: float) -> str:
        sign = "plus" if bias_V >= 0.0 else "minus"
        return f"{sign}_{abs(bias_V):0.3f}V"

    def _create_domain(self) -> mesh.Mesh:
        return mesh.create_interval(MPI.COMM_WORLD, self.config.num_cells, np.array([0.0, self.config.length_au], dtype=np.float64))

    def _create_doping_functions(self, domain: mesh.Mesh) -> tuple[fem.Function, fem.Function]:
        space = fem.functionspace(domain, ("Lagrange", 1))
        donor = fem.Function(space, name="donor_density")
        acceptor = fem.Function(space, name="acceptor_density")
        donor.interpolate(lambda x: self._donor_density_profile(x[0]))
        acceptor.interpolate(lambda x: self._acceptor_density_profile(x[0]))
        return donor, acceptor

    def _equilibrium_contacts(self) -> ContactValues:
        n_left = self.config.ni_mass_action_au**2 / self.config.NA_au
        p_left = self.config.NA_au
        n_right = self.config.ND_au
        p_right = self.config.ni_mass_action_au**2 / self.config.ND_au
        fermi_level = 0.5 * (
            self.config.Ec0_ha
            + self.config.kBT_ha * np.log(n_left / self.config.Nc_au)
            + self.config.Ev0_ha
            - self.config.kBT_ha * np.log(p_left / self.config.Nv_au)
        )
        phi_left = 0.5 * (
            self.config.Ec0_ha - fermi_level + self.config.kBT_ha * np.log(n_left / self.config.Nc_au)
            + self.config.Ev0_ha - fermi_level - self.config.kBT_ha * np.log(p_left / self.config.Nv_au)
        )
        phi_right = 0.5 * (
            self.config.Ec0_ha - fermi_level + self.config.kBT_ha * np.log(n_right / self.config.Nc_au)
            + self.config.Ev0_ha - fermi_level - self.config.kBT_ha * np.log(p_right / self.config.Nv_au)
        )
        return ContactValues(phi_left, phi_right, fermi_level, fermi_level, fermi_level, fermi_level)

    def _biased_contacts(self, bias_V: float) -> ContactValues:
        equilibrium = self._equilibrium_contacts()
        bias_ha = bias_V / self.config.hartree_ev
        return ContactValues(
            equilibrium.phi_left_ha,
            equilibrium.phi_right_ha + bias_ha,
            equilibrium.Fn_left_ha,
            equilibrium.Fn_right_ha - bias_ha,
            equilibrium.Fp_left_ha,
            equilibrium.Fp_right_ha - bias_ha,
        )

    def _build_initial_solution(
        self,
        domain: mesh.Mesh,
        donor: fem.Function,
        acceptor: fem.Function,
        contacts: ContactValues,
        previous_state: BiasState | None,
    ) -> fem.Function:
        lagrange = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        mixed_space = fem.functionspace(domain, basix.ufl.mixed_element([lagrange, lagrange, lagrange]))
        solution = fem.Function(mixed_space, name="pn_solution")
        phi_space, phi_map = mixed_space.sub(0).collapse()
        fn_space, fn_map = mixed_space.sub(1).collapse()
        fp_space, fp_map = mixed_space.sub(2).collapse()

        if previous_state is None:
            equilibrium_phi = self._solve_equilibrium_poisson(domain, donor, acceptor, contacts)
            coordinates = fn_space.tabulate_dof_coordinates()[:, 0]
            solution.x.array[phi_map] = equilibrium_phi.x.array
            solution.x.array[fn_map] = np.interp(coordinates, [0.0, coordinates.max()], [contacts.Fn_left_ha, contacts.Fn_right_ha])
            solution.x.array[fp_map] = np.interp(coordinates, [0.0, coordinates.max()], [contacts.Fp_left_ha, contacts.Fp_right_ha])
        else:
            solution.x.array[:] = previous_state.solution.x.array
            old_contacts = previous_state.contacts
            for coordinates, dof_map, new_left, new_right, old_left, old_right in (
                (phi_space.tabulate_dof_coordinates()[:, 0], phi_map, contacts.phi_left_ha, contacts.phi_right_ha, old_contacts.phi_left_ha, old_contacts.phi_right_ha),
                (fn_space.tabulate_dof_coordinates()[:, 0], fn_map, contacts.Fn_left_ha, contacts.Fn_right_ha, old_contacts.Fn_left_ha, old_contacts.Fn_right_ha),
                (fp_space.tabulate_dof_coordinates()[:, 0], fp_map, contacts.Fp_left_ha, contacts.Fp_right_ha, old_contacts.Fp_left_ha, old_contacts.Fp_right_ha),
            ):
                solution.x.array[dof_map] += np.interp(coordinates, [0.0, coordinates.max()], [new_left - old_left, new_right - old_right])

        solution.x.scatter_forward()
        return solution

    def _solve_equilibrium_poisson(
        self,
        domain: mesh.Mesh,
        donor: fem.Function,
        acceptor: fem.Function,
        contacts: ContactValues,
    ) -> fem.Function:
        space = fem.functionspace(domain, ("Lagrange", 1))
        phi = fem.Function(space, name="equilibrium_phi")
        coordinates = space.tabulate_dof_coordinates()[:, 0]
        phi.x.array[:] = np.interp(coordinates, [0.0, coordinates.max()], [contacts.phi_left_ha, contacts.phi_right_ha])
        phi.x.scatter_forward()

        left_facets, right_facets = self._locate_contact_facets(domain)
        facet_dim = domain.topology.dim - 1
        left_dofs = fem.locate_dofs_topological(space, facet_dim, left_facets)
        right_dofs = fem.locate_dofs_topological(space, facet_dim, right_facets)
        test = ufl.TestFunction(space)
        n = self._electron_density_ufl(phi, PETSc.ScalarType(contacts.Fn_left_ha))
        p = self._hole_density_ufl(phi, PETSc.ScalarType(contacts.Fp_left_ha))
        residual = (
            self.config.epsilon_r * ufl.dot(ufl.grad(phi), ufl.grad(test))
            - self.config.area_au * (p - n + donor - acceptor) * test
        ) * ufl.dx
        problem = NonlinearProblem(
            residual,
            phi,
            bcs=[
                fem.dirichletbc(PETSc.ScalarType(contacts.phi_left_ha), left_dofs, space),
                fem.dirichletbc(PETSc.ScalarType(contacts.phi_right_ha), right_dofs, space),
            ],
            petsc_options_prefix="pn_equilibrium_",
            petsc_options=self._nonlinear_solver_options(),
        )
        problem.solve()
        phi.x.scatter_forward()
        return phi

    def _boundary_conditions(
        self,
        mixed_space,
        contacts: ContactValues,
        left_facets: np.ndarray,
        right_facets: np.ndarray,
    ) -> list[fem.DirichletBC]:
        conditions: list[fem.DirichletBC] = []
        for subspace_index, left_value, right_value, name in (
            (0, contacts.phi_left_ha, contacts.phi_right_ha, "phi"),
            (1, contacts.Fn_left_ha, contacts.Fn_right_ha, "Fn"),
            (2, contacts.Fp_left_ha, contacts.Fp_right_ha, "Fp"),
        ):
            collapsed_space, _ = mixed_space.sub(subspace_index).collapse()
            left_field = self._constant_function(collapsed_space, left_value, f"{name}_left")
            right_field = self._constant_function(collapsed_space, right_value, f"{name}_right")
            left_dofs = fem.locate_dofs_topological((mixed_space.sub(subspace_index), collapsed_space), mixed_space.mesh.topology.dim - 1, left_facets)
            right_dofs = fem.locate_dofs_topological((mixed_space.sub(subspace_index), collapsed_space), mixed_space.mesh.topology.dim - 1, right_facets)
            conditions.append(fem.dirichletbc(left_field, left_dofs, mixed_space.sub(subspace_index)))
            conditions.append(fem.dirichletbc(right_field, right_dofs, mixed_space.sub(subspace_index)))
        return conditions

    def _constant_function(self, space, value: float, name: str) -> fem.Function:
        field = fem.Function(space, name=name)
        field.x.array[:] = value
        field.x.scatter_forward()
        return field

    def _locate_contact_facets(self, domain: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
        facet_dim = domain.topology.dim - 1
        left_facets = mesh.locate_entities_boundary(domain, facet_dim, lambda x: np.isclose(x[0], 0.0))
        right_facets = mesh.locate_entities_boundary(domain, facet_dim, lambda x: np.isclose(x[0], domain.geometry.x[:, 0].max()))
        return left_facets, right_facets

    def _extract_solution_components(self, state: BiasState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi_field = state.solution.sub(0).collapse()
        fn_field = state.solution.sub(1).collapse()
        fp_field = state.solution.sub(2).collapse()
        x_au, phi_ha = self._sorted_field_values(phi_field)
        _, Fn_ha = self._sorted_field_values(fn_field)
        _, Fp_ha = self._sorted_field_values(fp_field)
        return x_au, phi_ha, Fn_ha, Fp_ha

    def _sorted_field_values(self, field: fem.Function) -> tuple[np.ndarray, np.ndarray]:
        coordinates = field.function_space.tabulate_dof_coordinates()[:, 0]
        order = np.argsort(coordinates)
        return coordinates[order], field.x.array.real[order]

    def _nonlinear_solver_options(self) -> dict[str, object]:
        return {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_damping": self.config.line_search_damping,
            "snes_rtol": self.config.newton_rtol,
            "snes_atol": self.config.newton_atol,
            "snes_max_it": self.config.newton_max_it,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_shift_type": "nonzero",
            "pc_factor_shift_amount": 1.0e-12,
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        }

    def _donor_density_profile(self, x_au: np.ndarray) -> np.ndarray:
        transition = np.tanh((x_au - 0.5 * self.config.length_au) / self.config.transition_width_au)
        return self.config.ND_au * 0.5 * (1.0 + transition)

    def _acceptor_density_profile(self, x_au: np.ndarray) -> np.ndarray:
        transition = np.tanh((x_au - 0.5 * self.config.length_au) / self.config.transition_width_au)
        return self.config.NA_au * 0.5 * (1.0 - transition)

    def _electron_density_ufl(self, phi, Fn):
        argument = (Fn + phi - self.config.Ec0_ha) / self.config.kBT_ha
        return self.config.Nc_au * ufl.exp(self._clip_argument(argument))

    def _hole_density_ufl(self, phi, Fp):
        argument = (self.config.Ev0_ha - phi - Fp) / self.config.kBT_ha
        return self.config.Nv_au * ufl.exp(self._clip_argument(argument))

    def _recombination_ufl(self, n, p):
        return self.config.gamma_au * (n * p - self.config.ni_mass_action_au**2)

    def _clip_argument(self, argument):
        return ufl.max_value(ufl.min_value(argument, self.config.exponential_clip), -self.config.exponential_clip)

    def _electron_density_numpy(self, phi_ha: np.ndarray, Fn_ha: np.ndarray) -> np.ndarray:
        argument = np.clip((Fn_ha + phi_ha - self.config.Ec0_ha) / self.config.kBT_ha, -self.config.exponential_clip, self.config.exponential_clip)
        return self.config.Nc_au * np.exp(argument)

    def _hole_density_numpy(self, phi_ha: np.ndarray, Fp_ha: np.ndarray) -> np.ndarray:
        argument = np.clip((self.config.Ev0_ha - phi_ha - Fp_ha) / self.config.kBT_ha, -self.config.exponential_clip, self.config.exponential_clip)
        return self.config.Nv_au * np.exp(argument)

    def _density_to_m3(self, density_au: np.ndarray) -> np.ndarray:
        return density_au / (self.config.bohr_radius_m**3)

    def _length_to_um(self, length_au: np.ndarray) -> np.ndarray:
        return 1.0e6 * length_au * self.config.bohr_radius_m

    def _current_to_amp(self, current_au: np.ndarray) -> np.ndarray:
        return current_au * self.config.elementary_charge_c / self.config.atomic_time_s
