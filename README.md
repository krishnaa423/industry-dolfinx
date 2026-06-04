# DOLFINx PDE Gallery

Finite-element notes and examples for portfolio-scale scientific computing.

## Result

The main example solves Poisson's equation

```math
-\partial_i(k \partial_i u) = f \quad \text{in } \Omega,\qquad u=g \quad \text{on } \partial\Omega
```

with the weak form

```math
\int_\Omega k\,\partial_i u\,\partial_i v\,dx = \int_\Omega f v\,dx.
```

The script in `src/poisson.py` is written for DOLFINx and PyVista. When run in
a DOLFINx-enabled environment it writes a solution snapshot to `results/`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install dolfinx pyvista
python src/poisson.py
```

## Portfolio notes

This project demonstrates weak-form derivation, mesh construction, boundary
conditions, sparse linear solves, and scientific visualization. The deeper
derivation lives in `docs/main.tex`.
