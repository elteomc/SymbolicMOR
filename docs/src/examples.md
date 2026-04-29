# Examples

## Cubic Decay

The minimal nontrivial lift is a scalar cubic term:

```julia
using SymbolicMOR
using Symbolics

@variables x y
ls = lift_system([x, y], [-x^3, -y])
```

The lifted system introduces one auxiliary variable, typically equivalent to
`w1 = x^2`, so `dx/dt = -x^3` becomes `dx/dt = -w1*x`.

Run the benchmark with:

```bash
julia --project=. benchmarks/cubic_decay_benchmark.jl
```

## Van der Pol

The Van der Pol benchmark checks a mixed cubic term of the form `x^2*y`:

```bash
julia --project=. benchmarks/vanderpol_benchmark.jl
```

## Lorenz

Lorenz is already quadratic, so it is a useful baseline for operator extraction,
POD, Galerkin projection, and snapshot timing.

```bash
julia --project=. benchmarks/lorenz_benchmark.jl
```

## Allen-Cahn Reaction Smoke Test

`benchmarks/allen_cahn_benchmark.jl` keeps a conservative pure cubic reaction
smoke test. The affine form `u - u^3` is documented as a current symbolic
simplification limitation, because it exposed a long-running lift path during
development.

## Quadratic Operators

The dense compatibility path still returns `(A, H, c)`:

```julia
A, H, c = extract_operators(ls)
```

For larger sparse quadratic systems, prefer the tensor path:

```julia
A, Q, c = extract_quadratic_tensor(ls)
out = zeros(length(c))
evaluate_quadratic!(out, Q, randn(length(c)))
```

Run the dense-vs-tensor benchmark with:

```bash
julia --project=. benchmarks/quadratic_operator_benchmark.jl
```
