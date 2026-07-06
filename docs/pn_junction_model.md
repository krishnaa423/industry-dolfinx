# 1D Silicon p-n Junction Model

## Unknowns

The simulator solves for three fields on a 1D interval:

- `\phi(x)` : electrostatic potential in volts
- `F_n(x)` : electron quasi-Fermi energy in joules
- `F_p(x)` : hole quasi-Fermi energy in joules

## Carrier densities

Boltzmann carrier densities are used:

- `n(\phi, F_n) = N_C \exp((F_n - E_{C0} + q\phi)/(k_B T))`
- `p(\phi, F_p) = N_V \exp((E_{V0} - q\phi - F_p)/(k_B T))`

To keep the exponentials numerically stable in SI units, the exponent argument
is clipped to `[-80, 80]` in the UFL expressions.

## Poisson equation

The electrostatic equation is

- `-\partial_x(\epsilon \partial_x \phi) = q[p - n + N_D^+ - N_A^-]`

The weak form used in the code is

- `\int \epsilon \phi' v_\phi' dx - \int q[p - n + N_D^+ - N_A^-] v_\phi dx = 0`

At equilibrium, `F_n = F_p = F_0`, so only `\phi(x)` is solved for in the
first stage.

## Continuity equations

The mixed drift-diffusion stage uses

- `J_n = \mu_n n \partial_x F_n`
- `J_p = \mu_p p \partial_x F_p`

and the weak residuals

- `\int \mu_n n F_n' v_n' dx + \int q U v_n dx = 0`
- `\int \mu_p p F_p' v_p' dx - \int q U v_p dx = 0`

The opposite recombination signs reflect electron and hole continuity.

## Recombination model

A simple toy recombination law is used:

- `U = \gamma (np - n_i^2)`

where

- `n_i = \sqrt{N_C N_V} \exp(-E_g/(2 k_B T))`

The implementation starts with `\gamma = 0` for the first bias ramp. Small
nonzero values such as `1e-16 m^3/s` can be explored later.

## Doping profile

The 1D device spans `L = 2e-6 m` and uses a smooth hyperbolic-tangent junction:

- `N_D^+ = N_D (1 + tanh((x - L/2)/w))/2`
- `N_A^- = N_A (1 - tanh((x - L/2)/w))/2`

with

- `N_A = N_D = 1e23 m^{-3}`
- `w = 2e-8 m`

This gives a p-type left half and an n-type right half without a sharp
discontinuity.

## Boundary conditions

### Equilibrium

At equilibrium,

- `F_n = F_p = F_0`

and Dirichlet values for `\phi` are estimated by local charge neutrality:

- on the p-contact, `p \approx N_A`
- on the n-contact, `n \approx N_D`

These formulas determine `\phi_L` and `\phi_R`.

### Bias ramp

For applied bias `V_app`, the contact conditions are

- `F_{n,L} = F_0`
- `F_{p,L} = F_0`
- `F_{n,R} = F_0 - q V_app`
- `F_{p,R} = F_0 - q V_app`

and the electrostatic boundary values are

- `\phi_L = \phi_{\mathrm{eq},L}`
- `\phi_R = \phi_{\mathrm{eq},R} + V_app`

## Bias ramp strategy

The forward-bias list is

- `0.0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6` volts

The previous converged mixed solution is reused as the initial guess for the
next bias point. This continuation strategy is important because the coupled
system is stiff in SI units.
