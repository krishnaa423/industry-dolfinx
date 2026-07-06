from __future__ import annotations

from mpi4py import MPI

from .device_config import DeviceConfig
from .simulation_runner import SimulationRunner


def main() -> None:
    summary = SimulationRunner(DeviceConfig()).run()
    if MPI.COMM_WORLD.rank == 0:
        print(f"Wrote outputs to {summary.docs_dir}")


if __name__ == "__main__":
    main()
