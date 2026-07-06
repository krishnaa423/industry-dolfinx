from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DeviceConfig:
    bohr_radius_m: float = 5.29177210903e-11
    hartree_ev: float = 27.211386245988
    atomic_time_s: float = 2.4188843265857e-17
    elementary_charge_c: float = 1.602176634e-19
    boltzmann_hartree_per_K: float = 3.166811563e-6
    temperature_K: float = 300.0
    epsilon_r: float = 11.7
    Ec0_ha: float = 1.12 / 27.211386245988
    Ev0_ha: float = 0.0
    Nc_m3: float = 2.8e25
    Nv_m3: float = 1.04e25
    ni_m3: float = 1.45e16
    NA_m3: float = 1.0e23
    ND_m3: float = 1.0e23
    gamma_m3_per_s: float = 1.0e-20
    length_m: float = 2.0e-6
    area_um2: float = 1.0
    transition_width_m: float = 2.0e-8
    num_cells: int = 200
    mu_n_scale: float = 1.0
    mu_p_scale: float = 0.048 / 0.135
    applied_biases_V: tuple[float, ...] = (
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    )
    newton_atol: float = 1.0e-10
    newton_rtol: float = 1.0e-8
    newton_max_it: int = 80
    line_search_damping: float = 0.5
    exponential_clip: float = 80.0
    docs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "docs")

    @property
    def kBT_ha(self) -> float:
        return self.boltzmann_hartree_per_K * self.temperature_K

    @property
    def length_au(self) -> float:
        return self.length_m / self.bohr_radius_m

    @property
    def area_au(self) -> float:
        return (self.area_um2 * 1.0e-12) / (self.bohr_radius_m**2)

    @property
    def transition_width_au(self) -> float:
        return self.transition_width_m / self.bohr_radius_m

    @property
    def Nc_au(self) -> float:
        return self.Nc_m3 * (self.bohr_radius_m**3)

    @property
    def Nv_au(self) -> float:
        return self.Nv_m3 * (self.bohr_radius_m**3)

    @property
    def ni_au(self) -> float:
        return self.ni_m3 * (self.bohr_radius_m**3)

    @property
    def ni_mass_action_au(self) -> float:
        return math.sqrt(self.Nc_au * self.Nv_au) * math.exp((self.Ev0_ha - self.Ec0_ha) / (2.0 * self.kBT_ha))

    @property
    def NA_au(self) -> float:
        return self.NA_m3 * (self.bohr_radius_m**3)

    @property
    def ND_au(self) -> float:
        return self.ND_m3 * (self.bohr_radius_m**3)

    @property
    def gamma_au(self) -> float:
        return self.gamma_m3_per_s * self.atomic_time_s / (self.bohr_radius_m**3)
