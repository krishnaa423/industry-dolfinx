# Implementation Notes

## Solver sequence

The implementation follows the requested staged build:

1. Equilibrium Poisson only
2. Mixed zero-bias solve with `gamma = 0`
3. Forward-bias ramp using continuation
4. I-V postprocessing

This keeps the hardest nonlinear solve from being the very first thing the code
attempts.

## SI units and stiffness

The simulator uses SI units throughout. That keeps the physical interpretation
clean, but it also makes the nonlinear system stiff because the exponentials in
the Boltzmann densities can grow very quickly.

To keep the first version stable:

- the exponential argument is clipped to `[-80, 80]`
- Newton solves use direct LU preconditioning through PETSc
- the Newton relaxation parameter is reduced from `1.0` to `0.6`

## Currents

The weak forms use

- `J_n = \mu_n n \partial_x F_n`
- `J_p = \mu_p p \partial_x F_p`

For plotting and the I-V curve, the code evaluates the solved fields at the CG1
nodes and estimates the gradients with a 1D finite-difference postprocessing
step. That is a simple first version and is sufficient for checking whether the
currents are roughly constant when `gamma = 0`.

## Expected checks

The main sanity checks are:

- left side behaves p-type and right side behaves n-type
- equilibrium potential bends smoothly across the junction
- far from the junction, majority carriers track the dopants
- at zero bias with `gamma = 0`, `F_n` and `F_p` stay nearly flat
- under forward bias, total current increases strongly

## Next improvements

Natural next refinements would be:

- add automated tests for the equilibrium and bias-ramp summaries
- improve current evaluation with direct finite-element projection
- explore nondimensionalization for stronger nonlinear robustness
- add reverse-bias runs after the forward-bias branch is stable
