from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _x_um(x_m: np.ndarray) -> np.ndarray:
    return 1.0e6 * x_m


def plot_doping_profile(
    x_m: np.ndarray,
    donor_m3: np.ndarray,
    acceptor_m3: np.ndarray,
    net_m3: np.ndarray,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(_x_um(x_m), donor_m3, label=r"$N_D^+$")
    axis.plot(_x_um(x_m), -acceptor_m3, label=r"$-N_A^-$")
    axis.plot(_x_um(x_m), net_m3, label=r"$N_D^+ - N_A^-$", linewidth=2.0)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("x (um)")
    axis.set_ylabel("doping (m$^{-3}$)")
    axis.set_title("Smooth p-n junction doping profile")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_equilibrium_summary(
    observables: dict[str, np.ndarray],
    donor_m3: np.ndarray,
    acceptor_m3: np.ndarray,
    net_m3: np.ndarray,
    destination: Path,
) -> None:
    x_um = _x_um(observables["x_m"])
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(x_um, observables["phi_V"])
    axes[0].set_ylabel(r"$\phi$ (V)")
    axes[0].set_title("Equilibrium Poisson solution")
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(x_um, np.maximum(observables["n_m3"], 1.0), label="n")
    axes[1].semilogy(x_um, np.maximum(observables["p_m3"], 1.0), label="p")
    axes[1].semilogy(x_um, np.maximum(donor_m3, 1.0), "--", label=r"$N_D^+$")
    axes[1].semilogy(x_um, np.maximum(acceptor_m3, 1.0), "--", label=r"$N_A^-$")
    axes[1].set_ylabel(r"density (m$^{-3}$)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(x_um, net_m3, label="net doping")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel(r"m$^{-3}$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_mixed_zero_bias(
    observables: dict[str, np.ndarray],
    destination: Path,
) -> None:
    x_um = _x_um(observables["x_m"])
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x_um, observables["phi_V"])
    axes[0].set_ylabel(r"$\phi$ (V)")
    axes[0].set_title("Mixed zero-bias solve")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_um, observables["Fn_J"], label=r"$F_n$")
    axes[1].plot(x_um, observables["Fp_J"], label=r"$F_p$")
    axes[1].set_ylabel("energy (J)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(x_um, observables["Jn_Apm2"], label=r"$J_n$")
    axes[2].plot(x_um, observables["Jp_Apm2"], label=r"$J_p$")
    axes[2].plot(x_um, observables["Jtot_Apm2"], label=r"$J_n + J_p$", linewidth=2.0)
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("current")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_bias_state(
    observables: dict[str, np.ndarray],
    bias_V: float,
    destination: Path,
) -> None:
    x_um = _x_um(observables["x_m"])
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)

    axes[0, 0].plot(x_um, observables["phi_V"])
    axes[0, 0].set_ylabel(r"$\phi$ (V)")
    axes[0, 0].set_title(f"Bias = {bias_V:.3f} V")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(x_um, observables["Fn_J"], label=r"$F_n$")
    axes[0, 1].plot(x_um, observables["Fp_J"], label=r"$F_p$")
    axes[0, 1].set_ylabel("energy (J)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].semilogy(x_um, np.maximum(observables["n_m3"], 1.0), label="n")
    axes[1, 0].semilogy(x_um, np.maximum(observables["p_m3"], 1.0), label="p")
    axes[1, 0].set_ylabel(r"density (m$^{-3}$)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(x_um, observables["Jn_Apm2"], label=r"$J_n$")
    axes[1, 1].plot(x_um, observables["Jp_Apm2"], label=r"$J_p$")
    axes[1, 1].set_ylabel("carrier current")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    axes[2, 0].plot(x_um, observables["Jtot_Apm2"], label=r"$J_{\mathrm{tot}}$")
    axes[2, 0].set_xlabel("x (um)")
    axes[2, 0].set_ylabel("total current")
    axes[2, 0].grid(True, alpha=0.25)
    axes[2, 0].legend()

    axes[2, 1].plot(x_um, observables["Jn_Apm2"] - observables["Jn_Apm2"][0], label=r"$J_n - J_n(0)$")
    axes[2, 1].plot(x_um, observables["Jp_Apm2"] - observables["Jp_Apm2"][0], label=r"$J_p - J_p(0)$")
    axes[2, 1].set_xlabel("x (um)")
    axes[2, 1].set_ylabel("current drift")
    axes[2, 1].grid(True, alpha=0.25)
    axes[2, 1].legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_iv_curve(
    biases_V: list[float],
    total_current_Apm2: list[float],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(biases_V, total_current_Apm2, marker="o")
    axis.set_xlabel("applied bias (V)")
    axis.set_ylabel(r"$J_{\mathrm{tot}}$ at right contact")
    axis.set_title("I-V curve from bias ramp")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
