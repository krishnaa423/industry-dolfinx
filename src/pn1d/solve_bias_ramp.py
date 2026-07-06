from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mpi4py import MPI

from .constants import BiasRampParameters, DeviceParameters, SiliconParameters, SolverParameters
from .mesh import ensure_output_dir
from .physics import biased_contacts
from .plotting import plot_bias_state, plot_iv_curve, write_json
from .solve_equilibrium import run_equilibrium, solve_mixed_state, split_mixed_state
from .physics import compute_observables


def run_bias_ramp(
    silicon: SiliconParameters | None = None,
    device: DeviceParameters | None = None,
    solver_params: SolverParameters | None = None,
    ramp: BiasRampParameters | None = None,
) -> dict[str, object]:
    silicon = silicon or SiliconParameters()
    device = device or DeviceParameters()
    solver_params = solver_params or SolverParameters()
    ramp = ramp or BiasRampParameters()

    equilibrium = run_equilibrium(silicon=silicon, device=device, solver_params=solver_params)
    output_dir = ensure_output_dir(solver_params, "bias_ramp")

    state = equilibrium.mixed_state
    total_currents: list[float] = []
    plot_paths: list[str] = []

    for bias_V in ramp.applied_bias_values_V:
        contacts = biased_contacts(silicon, equilibrium.contacts, bias_V)
        state = solve_mixed_state(
            domain=equilibrium.mesh,
            net_doping=equilibrium.net_doping,
            contacts=contacts,
            equilibrium_phi=equilibrium.equilibrium_potential,
            params=silicon,
            solver_params=solver_params,
            gamma_m3_per_s=ramp.gamma_m3_per_s,
            initial_state=state,
        )
        x_m, phi_values, Fn_values, Fp_values = split_mixed_state(state)
        observables = compute_observables(
            x_m=x_m,
            phi_V=phi_values,
            Fn_J=Fn_values,
            Fp_J=Fp_values,
            params=silicon,
            solver=solver_params,
        )
        total_currents.append(float(observables["Jtot_Apm2"][-1]))

        bias_tag = str(bias_V).replace(".", "p")
        plot_path = output_dir / f"bias_{bias_tag}_summary.png"
        plot_bias_state(observables, bias_V=bias_V, destination=plot_path)
        plot_paths.append(str(plot_path))

    iv_curve_path = output_dir / "iv_curve.png"
    plot_iv_curve(list(ramp.applied_bias_values_V), total_currents, iv_curve_path)

    summary = {
        **asdict(silicon),
        **asdict(device),
        **{**asdict(solver_params), "output_dir": str(solver_params.output_dir)},
        **asdict(ramp),
        "bias_plot_paths": plot_paths,
        "iv_curve_path": str(iv_curve_path),
        "right_contact_total_current_Apm2": total_currents,
    }
    write_json(output_dir / "bias_ramp_summary.json", summary)
    return summary


def main() -> None:
    summary = run_bias_ramp()
    if MPI.COMM_WORLD.rank == 0:
        print(f"Bias ramp outputs written to {Path(summary['iv_curve_path']).parent}")


if __name__ == "__main__":
    main()
