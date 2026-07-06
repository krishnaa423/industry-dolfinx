from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import fem, mesh


@dataclass(frozen=True)
class ContactValues:
    phi_left_ha: float
    phi_right_ha: float
    Fn_left_ha: float
    Fn_right_ha: float
    Fp_left_ha: float
    Fp_right_ha: float


@dataclass
class BiasState:
    domain: mesh.Mesh
    donor: fem.Function
    acceptor: fem.Function
    solution: fem.Function
    contacts: ContactValues
    converged: bool


@dataclass(frozen=True)
class BiasPlotData:
    bias_V: float
    converged: bool
    x_um: np.ndarray
    phi_ha: np.ndarray
    Fn_ha: np.ndarray
    Fp_ha: np.ndarray
    donor_m3: np.ndarray
    acceptor_m3: np.ndarray
    total_current_A: np.ndarray
