from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device_config import DeviceConfig
from .plot_writer import PlotWriter
from .pn_solver import PnJunctionSolver
from .simulation_data import BiasState


@dataclass(frozen=True)
class SimulationSummary:
    docs_dir: str
    iv_curve: str


class SimulationRunner:
    def __init__(self, config: DeviceConfig) -> None:
        self.config = config
        self.solver = PnJunctionSolver(config)
        self.plot_writer = PlotWriter(config.docs_dir)

    def run(self) -> SimulationSummary:
        state: BiasState | None = None
        results: list[dict[str, float | bool | str]] = []

        for bias_V in sorted(self.config.applied_biases_V, key=lambda value: (abs(value), value)):
            state = self.solver.solve_bias(bias_V, state)
            plot_data = self.solver.build_plot_data(state, bias_V)
            plot_name = f"bias_{self.solver.bias_label(bias_V)}.png"
            self.plot_writer.write_bias_plot(plot_data, plot_name)
            results.append(
                {
                    "bias_V": bias_V,
                    "right_contact_current_A": float(plot_data.total_current_A[-1]),
                    "current_min_A": float(np.min(plot_data.total_current_A)),
                    "current_max_A": float(np.max(plot_data.total_current_A)),
                    "nonlinear_converged": plot_data.converged,
                }
            )

        results.sort(key=lambda item: float(item["bias_V"]))
        self.plot_writer.write_iv_curve(
            [float(item["bias_V"]) for item in results],
            [float(item["right_contact_current_A"]) for item in results],
        )
        self.plot_writer.write_current_diagnostic(
            [float(item["bias_V"]) for item in results],
            [float(item["current_min_A"]) for item in results],
            [float(item["current_max_A"]) for item in results],
        )
        return SimulationSummary(
            docs_dir=str(self.config.docs_dir),
            iv_curve=str(self.config.docs_dir / "iv_curve.png"),
        )
