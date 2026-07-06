from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .simulation_data import BiasPlotData


class PlotWriter:
    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def write_bias_plot(self, plot_data: BiasPlotData, filename: str) -> None:
        figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        axes[0].plot(plot_data.x_um, plot_data.phi_ha, color="tab:blue")
        axes[0].set_ylabel("phi (Ha)")
        axes[0].set_title(f"Bias = {plot_data.bias_V:+.3f} V")

        axes[1].plot(plot_data.x_um, plot_data.Fn_ha, label="Fn", color="tab:green")
        axes[1].plot(plot_data.x_um, plot_data.Fp_ha, label="Fp", color="tab:red")
        axes[1].set_ylabel("energy (Ha)")
        axes[1].legend()

        axes[2].semilogy(plot_data.x_um, np.maximum(plot_data.donor_m3, 1.0), "--", label="ND", color="tab:purple")
        axes[2].semilogy(plot_data.x_um, np.maximum(plot_data.acceptor_m3, 1.0), "--", label="NA", color="tab:orange")
        axes[2].set_ylabel("density (m^-3)")
        axes[2].set_xlabel("x (um)")
        axes[2].legend()

        for axis in axes:
            axis.grid(True, alpha=0.25)

        figure.tight_layout()
        figure.savefig(self.docs_dir / filename, dpi=180)
        plt.close(figure)

    def write_iv_curve(self, biases_V: list[float], currents_A: list[float]) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(biases_V, currents_A, marker="o")
        axis.set_xlabel("bias (V)")
        axis.set_ylabel("right contact current (A)")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.docs_dir / "iv_curve.png", dpi=180)
        plt.close(figure)

    def write_current_diagnostic(self, biases_V: list[float], min_currents_A: list[float], max_currents_A: list[float]) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(biases_V, min_currents_A, marker="o", label="min I(x)")
        axis.plot(biases_V, max_currents_A, marker="o", label="max I(x)")
        axis.set_xlabel("bias (V)")
        axis.set_ylabel("current (A)")
        axis.legend()
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.docs_dir / "current_diagnostic.png", dpi=180)
        plt.close(figure)
