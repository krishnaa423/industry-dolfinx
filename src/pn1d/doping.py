from __future__ import annotations

import numpy as np
from dolfinx import fem, mesh

from .constants import DeviceParameters


def donor_density_numpy(x_m: np.ndarray, device: DeviceParameters) -> np.ndarray:
    transition = np.tanh((x_m - 0.5 * device.length_m) / device.junction_width_m)
    return device.ND_m3 * 0.5 * (1.0 + transition)


def acceptor_density_numpy(x_m: np.ndarray, device: DeviceParameters) -> np.ndarray:
    transition = np.tanh((x_m - 0.5 * device.length_m) / device.junction_width_m)
    return device.NA_m3 * 0.5 * (1.0 - transition)


def net_doping_numpy(x_m: np.ndarray, device: DeviceParameters) -> np.ndarray:
    return donor_density_numpy(x_m, device) - acceptor_density_numpy(x_m, device)


def interpolate_profile(
    domain: mesh.Mesh,
    space: fem.FunctionSpace,
    name: str,
    profile,
) -> fem.Function:
    field = fem.Function(space, name=name)
    field.interpolate(lambda x: profile(x[0]))
    return field


def build_doping_functions(
    domain: mesh.Mesh,
    space: fem.FunctionSpace,
    device: DeviceParameters,
) -> dict[str, fem.Function]:
    return {
        "donor": interpolate_profile(
            domain,
            space,
            "donor_density",
            lambda x: donor_density_numpy(x, device),
        ),
        "acceptor": interpolate_profile(
            domain,
            space,
            "acceptor_density",
            lambda x: acceptor_density_numpy(x, device),
        ),
        "net": interpolate_profile(
            domain,
            space,
            "net_doping",
            lambda x: net_doping_numpy(x, device),
        ),
    }
