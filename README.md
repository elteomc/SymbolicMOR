# SymbolicMOR.jl

SymbolicMOR.jl is a Julia prototype for **intrusive model order reduction** of nonlinear polynomial ODEs using a compact lift-and-learn workflow:

1. Symbolically lift polynomial dynamics into **quadratic** form (`lift_system`).
2. Simulate **snapshot** trajectories and build a **POD** basis (`generate_snapshots`, `compute_pod_basis`).
3. Extract dense quadratic operators **A**, **H**, **c** and **Galerkin-project** them (`extract_operators`, `galerkin_project`, `rom_rhs!`).
4. Compare **serial vs parallel** snapshot generation (`generate_snapshots_parallel`, `generate_snapshots_ensemble`, `benchmark_serial_vs_parallel`, aliased as `benchmark_scaling`).

It was developed as coursework for MIT **18.337 / 6.7320**, Parallel Computing and Scientific Machine Learning.

License: MIT. See `LICENSE`.

## Status

This is a **research prototype**, not a registered General registry package. It is suitable for reproducible demos, benchmarks, and course reports; SciML-grade polish (compat breadth, docs on Documenter.jl, MTK-first APIs, etc.) is left as future work.

## Installation

Clone and instantiate:

```bash
git clone https://github.com/elteomc/SymbolicMOR.git
cd SymbolicMOR
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Tests

```bash
julia --project=. test/runtests.jl
```

## Demo script

```bash
julia --project=. scripts/lorenz_demo.jl
```

This walks through symbolic lifting for Lorenz (already quadratic, zero auxiliary variables), snapshots, POD, and saves an SVD decay plot (`lorenz_svd.png`). For chaotic dynamics, interpret errors over **short horizons** or statistically; pointwise long-time trajectory matching is not the right metric.

## Benchmark scripts (`benchmarks/`)

Run from the repo root with `julia --project=. ...`. Suggested order:

| Script | Role |
| --- | --- |
| `benchmarks/cubic_decay_benchmark.jl` | Minimal cubic nonlinearity: verifies lifting and trajectory consistency vs the original ODE. |
| `benchmarks/vanderpol_benchmark.jl` | Mixed cubic term (`x^2*y`-type structure after expansion). |
| `benchmarks/lorenz_benchmark.jl` | Familiar quadratic chaos benchmark; POD + ROM projection spot-check + serial/parallel timing. |
| `benchmarks/allen_cahn_benchmark.jl` | Conservative Allen-Cahn reaction smoke benchmark (`du/dt = -u^3`) plus a documented affine reaction limitation. |

Parallel scaling smoke test (needs workers, e.g. `-p 4`):

```bash
julia --project=. -p 4 scripts/scaling_benchmark.jl
```

## Limitations

- Intended for **polynomial right-hand sides**; broader polynomialization of non-polynomial dynamics remains experimental (`polynomialize.jl`).
- Quadratic tensors use dense **`H * kron(u, u)`**: fine for demos, poor scaling for large lifted dimensions.
- Parallel snapshots use Julia **Distributed**; overhead dominates small ensembles or Windows setups. Linux/cluster runs are cleaner for scaling plots.
- Full SciML ecosystem alignment (e.g. **ModelingToolkit `ODESystem`** entry points, **EnsembleProblem** orchestration) is not implemented here yet.

## Repository layout

- `src/` : library sources (`lift/`, `learn/`, `scale/`, `SymbolicMOR.jl`)
- `scripts/` : Lorenz demo and scaling driver
- `benchmarks/` : benchmark scripts (see table above)
- `test/` : tests
- `Project.toml`

## Documentation

Build the local Documenter.jl site with:

```bash
julia --project=docs -e "using Pkg; Pkg.instantiate()"
julia --project=docs docs/make.jl
```

The generated site is written to `docs/build/`.

## References

- B. Kramer and K. E. Willcox, "Nonlinear model order reduction via lifting transformations and proper orthogonal decomposition," *AIAA Journal*, 2019.
- A. Bychkov and G. Pogudin, "Optimal monomial quadratization for ODE systems," arXiv:2103.08013, 2021.
