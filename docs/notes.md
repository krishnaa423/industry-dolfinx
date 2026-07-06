# Notes

This version of the simulator uses atomic units internally:

- length in `a0`
- energy and electrostatic potential in `Ha`
- time in `t0`
- densities in `a0^-3`

The public plots still use microns on the x-axis and report carrier densities in
`m^-3`, plus current density in `A/m^2` and area-integrated current in `A`.

## Equations

Unknowns:

- `phi(x)`
- `Fn(x)`
- `Fp(x)`

Boltzmann carrier densities:

- `n = Nc exp((Fn + phi - Ec0) / kBT)`
- `p = Nv exp((Ev0 - phi - Fp) / kBT)`

Recombination:

- `U = gamma (n p - ni^2)`

Weak forms used in the code:

- `int eps_r phi_x vphi_x dx - int A (p - n + ND - NA) vphi dx = 0`
- `int mu_n n Fn_x vn_x dx + int A U vn dx = 0`
- `-int mu_p p Fp_x vp_x dx + int A U vp dx = 0`

The implementation keeps `n`, `p`, `ND`, `NA`, and `ni` as 3D densities in
`a0^-3` and multiplies the source terms by the cross-sectional area `A` when
reducing to a 1D device model.

## Boundary conditions

The contacts are ideal Ohmic Dirichlet boundaries on all three unknowns.

Equilibrium contact densities are approximated with local charge neutrality:

- left p-contact: `pL ~= NA`, `nL ~= ni^2 / NA`
- right n-contact: `nR ~= ND`, `pR ~= ni^2 / ND`

Those contact densities are inverted through the Maxwell-Boltzmann formulas to
build the quasi-Fermi reference level and the equilibrium electrostatic contact
potentials.

For an applied bias `V`, the implementation uses:

- `phi_right <- phi_right_eq + V_Ha`
- `Fn_right <- F0 - V_Ha`
- `Fp_right <- F0 - V_Ha`

with the left contact held at its equilibrium reference values.

## Constants and assumptions

- silicon `eps_r = 11.7`
- `Eg = 1.12 eV`
- `Ec0 = Eg`, `Ev0 = 0`
- `Nc = 2.8e25 m^-3`
- `Nv = 1.04e25 m^-3`
- `ni = 1.45e16 m^-3`
- default `NA = ND = 1e23 m^-3`
- default area `A = 1 um^2`

The mobility handling is intentionally pragmatic for now. The code uses scaled
atomic-unit mobilities `mu_n = 1` and `mu_p = 0.048 / 0.135` so the nonlinear
problem is easier to continue across bias points while preserving the electron
to hole mobility ratio from the silicon inputs. The original SI values are
still recorded in the parameter metadata for later refinement.

## Outputs

The sweep writes:

- `docs/iv_curve.png`
- `docs/current_diagnostic.png`

At the moment the equilibrium point converges cleanly, while the biased states
are still useful continuation snapshots rather than fully converged nonlinear
solutions.
