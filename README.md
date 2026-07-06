# DOLFINx 1D Silicon p-n Junction

This project builds a minimal 1D silicon p-n junction drift-diffusion-Poisson
simulator in Python with DOLFINx.

The simulator is organized in steps:

1. Solve the equilibrium Poisson problem in SI units.
2. Lift that solution into a mixed finite-element solve for
   `(\phi, F_n, F_p)`.
3. Ramp forward bias gradually while reusing the previous converged state as the
   next initial guess.
4. Plot carrier densities, quasi-Fermi energies, currents, and an I-V curve.

## Environment

Use the existing Conda environment:

```bash
conda activate dolfinx
python -c "import dolfinx; print(dolfinx.__version__)"
```

This implementation was written against DOLFINx `0.11.0`.

## Run

Equilibrium plus zero-bias mixed solve:

```bash
conda activate dolfinx
python -m src.pn1d.solve_equilibrium
```

Bias ramp:

```bash
conda activate dolfinx
python -m src.pn1d.solve_bias_ramp
```

Full pipeline:

```bash
conda activate dolfinx
python -m src.pn1d.main
```

## Repository layout

- `src/pn1d/constants.py`
  Silicon constants, device parameters, bias values, and solver settings.
- `src/pn1d/mesh.py`
  1D interval mesh helpers and output-directory helpers.
- `src/pn1d/doping.py`
  Smooth donor/acceptor profiles for the p-n junction.
- `src/pn1d/physics.py`
  Density formulas, recombination model, contact conditions, and observables.
- `src/pn1d/solve_equilibrium.py`
  Equilibrium Poisson solve and mixed zero-bias solve.
- `src/pn1d/solve_bias_ramp.py`
  Forward-bias ramp and I-V curve generation.
- `src/pn1d/plotting.py`
  Matplotlib output helpers.
- `docs/pn_junction_model.md`
  Model equations and boundary conditions.
- `docs/implementation_notes.md`
  Practical notes about solver choices and current limitations.

## Model summary

Unknowns:

- `\phi(x)` in volts
- `F_n(x)` in joules
- `F_p(x)` in joules

Carrier densities use Boltzmann statistics:

- `n(\phi, F_n) = N_C exp((F_n - E_{C0} + q\phi)/(k_B T))`
- `p(\phi, F_p) = N_V exp((E_{V0} - q\phi - F_p)/(k_B T))`

The equilibrium solve keeps `F_n = F_p = F_0`.

The mixed solve uses a toy recombination model

- `U = \gamma (np - n_i^2)`

with `\gamma = 0` by default for the initial ramp.

## Output

Results are written under `results/`:

- `results/equilibrium/`
  - `doping_profile.png`
  - `equilibrium_summary.png`
  - `mixed_zero_bias.png`
- `results/bias_ramp/`
  - one summary plot per applied bias
  - `iv_curve.png`

The equilibrium plots should show:

- smooth band bending across the junction
- majority carriers matching dopants away from the interface
- flat quasi-Fermi levels in the zero-bias mixed solve

The bias-ramp plots should show:

- quasi-Fermi splitting under forward bias
- stronger total current at higher forward bias
- approximately spatially constant currents for `gamma = 0`
