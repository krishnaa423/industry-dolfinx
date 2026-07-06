from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SiliconParameters:
    q: float = 1.602176634e-19
    kB: float = 1.380649e-23
    eps0: float = 8.8541878128e-12
    T: float = 300.0
    eps_si: float = 11.7 * 8.8541878128e-12
    Nc: float = 2.8e25
    Nv: float = 1.04e25
    Eg: float = 1.12 * 1.602176634e-19
    Ec0: float = 0.0
    Ev0: float = -(1.12 * 1.602176634e-19)
    mu_n: float = 0.135
    mu_p: float = 0.048
    F0: float = 0.0

    @property
    def thermal_energy_J(self) -> float:
        return self.kB * self.T

    @property
    def thermal_voltage_V(self) -> float:
        return self.thermal_energy_J / self.q

    @property
    def ni(self) -> float:
        return float(np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2.0 * self.thermal_energy_J)))


@dataclass(frozen=True)
class DeviceParameters:
    length_m: float = 2.0e-6
    num_cells: int = 400
    NA_m3: float = 1.0e23
    ND_m3: float = 1.0e23
    junction_width_m: float = 2.0e-8


@dataclass(frozen=True)
class BiasRampParameters:
    gamma_m3_per_s: float = 0.0
    applied_bias_values_V: tuple[float, ...] = (
        0.0,
        0.025,
        0.05,
        0.075,
        0.1,
        0.2,
        0.4,
        0.6,
    )


@dataclass(frozen=True)
class SolverParameters:
    newton_atol: float = 1.0e-10
    newton_rtol: float = 1.0e-8
    newton_max_it: int = 50
    newton_relaxation: float = 0.6
    exponential_clip: float = 80.0
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "results"
    )
