# SymbolicMOR.jl

SymbolicMOR.jl is a small Julia prototype for symbolic lift-and-learn model
order reduction of nonlinear polynomial ODEs.

The package currently focuses on four pieces:

1. Symbolic quadratization of polynomial ODE right-hand sides.
2. Snapshot generation from ODE ensembles.
3. POD basis computation and Galerkin projection for quadratic systems.
4. Serial, distributed, and SciML ensemble snapshot workflows.

This is research-prototype code. It is useful for course demos and small
experiments, but it is not yet a registered Julia package.

## Installation

From a clone of the repository:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run tests with:

```bash
julia --project=. test/runtests.jl
```

## Quick Start

```julia
using SymbolicMOR
using Symbolics

@variables x y
ls = lift_system([x, y], [-x^3, -y])

A, H, c = extract_operators(ls)
```

For a complete end-to-end script, see `scripts/lorenz_demo.jl`.
