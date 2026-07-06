# DOLFINx 1D Silicon p-n Junction

This repo contains a small 1D steady-state silicon p-n junction solver built in
Python with DOLFINx. The code solves for:

- `phi(x)` : electrostatic potential
- `Fn(x)` : electron quasi-Fermi energy
- `Fp(x)` : hole quasi-Fermi energy

The source is organized by responsibility:

- `src/main.py` : entrypoint
- `src/device_config.py` : constants and simulation settings
- `src/simulation_data.py` : shared data classes
- `src/pn_solver.py` : nonlinear finite-element solve
- `src/plot_writer.py` : figure generation
- `src/simulation_runner.py` : bias sweep workflow

## Run

```bash
conda activate dolfinx
python -m src.main
```

## Model

The solver uses Maxwell-Boltzmann carrier statistics:

\[
n(\phi,F_n)=N_c\exp\left(\frac{F_n+\phi-E_{c0}}{k_B T}\right),
\qquad
p(\phi,F_p)=N_v\exp\left(\frac{E_{v0}-\phi-F_p}{k_B T}\right).
\]

The recombination model is

\[
U=\gamma (np-n_i^2).
\]

The 1D donor and acceptor profiles are smooth hyperbolic-tangent transitions:

\[
N_D(x)=\frac{N_D^\star}{2}\left(1+\tanh\left(\frac{x-L/2}{w}\right)\right),
\qquad
N_A(x)=\frac{N_A^\star}{2}\left(1-\tanh\left(\frac{x-L/2}{w}\right)\right).
\]

## Variational Form

The code solves the mixed weak system for `phi`, `Fn`, and `Fp`.

Poisson:

\[
\int_0^L \epsilon_r\, \phi_x\, v_{\phi,x}\, dx
\;-\;
\int_0^L A (p-n+N_D-N_A)\, v_\phi\, dx
=0.
\]

Electron continuity:

\[
\int_0^L \mu_n\, n\, F_{n,x}\, v_{n,x}\, dx
\;+\;
\int_0^L A\, U\, v_n\, dx
=0.
\]

Hole continuity:

\[
-\int_0^L \mu_p\, p\, F_{p,x}\, v_{p,x}\, dx
\;+\;
\int_0^L A\, U\, v_p\, dx
=0.
\]

The solver uses first-order Lagrange elements and a fully coupled Newton solve
through PETSc SNES.

## Boundary Conditions

The contacts are ideal Ohmic Dirichlet boundaries for all three unknowns.

At the left p-contact:

\[
p_L \approx N_A^\star,
\qquad
n_L \approx \frac{n_i^2}{N_A^\star}.
\]

At the right n-contact:

\[
n_R \approx N_D^\star,
\qquad
p_R \approx \frac{n_i^2}{N_D^\star}.
\]

For applied bias `V`, the code uses the convention

\[
\phi_R=\phi_{R,\mathrm{eq}}+\frac{V}{E_h},
\qquad
F_{n,R}=F_0-\frac{V}{E_h},
\qquad
F_{p,R}=F_0-\frac{V}{E_h},
\]

while the left contact stays at its equilibrium value.

## Constants

The current defaults are:

- `T = 300 K`
- `eps_r = 11.7`
- `Ec0 = 1.12 / 27.211386245988 Ha`
- `Ev0 = 0`
- `Nc = 2.8e25 m^-3`
- `Nv = 1.04e25 m^-3`
- `ni = 1.45e16 m^-3`
- `NA = ND = 1.0e23 m^-3`
- `gamma = 1.0e-20 m^3/s`
- `L = 2.0e-6 m`
- `A = 1.0 um^2`
- `w = 2.0e-8 m`
- `num_cells = 200`

The code uses atomic units internally and converts positions and doping back to
microns and `m^-3` for plotting.

## Solve Procedure

The bias sweep is:

- `-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 V`

The actual solve order is by increasing absolute bias so continuation is easier.
The first step solves an equilibrium Poisson problem to initialize `phi`. Later
bias points reuse the previous mixed solution as the initial guess.

## Figures

The code writes:

- `docs/bias_plus_0.100V.png` style bias figures
- `docs/iv_curve.png`
- `docs/current_diagnostic.png`

Each bias figure contains:

- `phi(x)`
- `Fn(x)` and `Fp(x)`
- `ND(x)` and `NA(x)`

## Theory Note

The fuller derivation and discussion live in:

- `docs/theory/main.tex`
- `docs/theory/main.pdf`
