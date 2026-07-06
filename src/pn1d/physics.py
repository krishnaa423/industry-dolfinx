from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl

from .constants import DeviceParameters, SiliconParameters, SolverParameters


@dataclass(frozen=True)
class ContactState:
    phi_left_V: float
    phi_right_V: float
    Fn_left_J: float
    Fn_right_J: float
    Fp_left_J: float
    Fp_right_J: float


def clip_eta_ufl(eta, solver: SolverParameters):
    return ufl.max_value(
        ufl.min_value(eta, solver.exponential_clip),
        -solver.exponential_clip,
    )


def electron_density_ufl(phi, Fn, params: SiliconParameters, solver: SolverParameters):
    eta = (Fn - params.Ec0 + params.q * phi) / params.thermal_energy_J
    return params.Nc * ufl.exp(clip_eta_ufl(eta, solver))


def hole_density_ufl(phi, Fp, params: SiliconParameters, solver: SolverParameters):
    eta = (params.Ev0 - params.q * phi - Fp) / params.thermal_energy_J
    return params.Nv * ufl.exp(clip_eta_ufl(eta, solver))


def recombination_ufl(phi, Fn, Fp, params: SiliconParameters, solver: SolverParameters, gamma_m3_per_s: float):
    n = electron_density_ufl(phi, Fn, params, solver)
    p = hole_density_ufl(phi, Fp, params, solver)
    return gamma_m3_per_s * (n * p - params.ni**2)


def electron_density_numpy(
    phi_V: np.ndarray,
    Fn_J: np.ndarray,
    params: SiliconParameters,
    solver: SolverParameters,
) -> np.ndarray:
    eta = np.clip((Fn_J - params.Ec0 + params.q * phi_V) / params.thermal_energy_J, -solver.exponential_clip, solver.exponential_clip)
    return params.Nc * np.exp(eta)


def hole_density_numpy(
    phi_V: np.ndarray,
    Fp_J: np.ndarray,
    params: SiliconParameters,
    solver: SolverParameters,
) -> np.ndarray:
    eta = np.clip((params.Ev0 - params.q * phi_V - Fp_J) / params.thermal_energy_J, -solver.exponential_clip, solver.exponential_clip)
    return params.Nv * np.exp(eta)


def contact_potential_p_side(params: SiliconParameters, device: DeviceParameters) -> float:
    return (params.Ev0 - params.F0 - params.thermal_energy_J * np.log(device.NA_m3 / params.Nv)) / params.q


def contact_potential_n_side(params: SiliconParameters, device: DeviceParameters) -> float:
    return (params.thermal_energy_J * np.log(device.ND_m3 / params.Nc) - params.F0 + params.Ec0) / params.q


def equilibrium_contacts(params: SiliconParameters, device: DeviceParameters) -> ContactState:
    phi_left = contact_potential_p_side(params, device)
    phi_right = contact_potential_n_side(params, device)
    return ContactState(
        phi_left_V=phi_left,
        phi_right_V=phi_right,
        Fn_left_J=params.F0,
        Fn_right_J=params.F0,
        Fp_left_J=params.F0,
        Fp_right_J=params.F0,
    )


def biased_contacts(
    params: SiliconParameters,
    equilibrium: ContactState,
    applied_bias_V: float,
) -> ContactState:
    delta_F = params.q * applied_bias_V
    return ContactState(
        phi_left_V=equilibrium.phi_left_V,
        phi_right_V=equilibrium.phi_right_V + applied_bias_V,
        Fn_left_J=params.F0,
        Fn_right_J=params.F0 - delta_F,
        Fp_left_J=params.F0,
        Fp_right_J=params.F0 - delta_F,
    )


def compute_observables(
    x_m: np.ndarray,
    phi_V: np.ndarray,
    Fn_J: np.ndarray,
    Fp_J: np.ndarray,
    params: SiliconParameters,
    solver: SolverParameters,
) -> dict[str, np.ndarray]:
    n = electron_density_numpy(phi_V, Fn_J, params, solver)
    p = hole_density_numpy(phi_V, Fp_J, params, solver)
    dFn_dx = np.gradient(Fn_J, x_m, edge_order=2)
    dFp_dx = np.gradient(Fp_J, x_m, edge_order=2)
    Jn = params.mu_n * n * dFn_dx
    Jp = params.mu_p * p * dFp_dx
    return {
        "x_m": x_m,
        "phi_V": phi_V,
        "Fn_J": Fn_J,
        "Fp_J": Fp_J,
        "n_m3": n,
        "p_m3": p,
        "Jn_Apm2": Jn,
        "Jp_Apm2": Jp,
        "Jtot_Apm2": Jn + Jp,
    }
