# SymbolicMOR.jl

SymbolicMOR.jl is a Julia prototype for **intrusive model order reduction** of nonlinear polynomial ODEs using a compact lift-and-learn workflow:

1. Symbolically lift polynomial dynamics into **quadratic** form (`lift_system`), either from vectors of symbolic variables/RHS expressions or from a narrow explicit ModelingToolkit `ODESystem`.
2. Simulate **snapshot** trajectories and build a **POD** basis (`generate_snapshots`, `compute_pod_basis`).
3. Extract dense, sparse, or tensor quadratic operators **A**, **H/Q**, **c** and **Galerkin-project** them (`extract_operators`, `extract_quadratic_tensor`, `galerkin_project`, `rom_rhs!`).
4. Compare **serial vs parallel** snapshot generation (`generate_snapshots_parallel`, `generate_snapshots_ensemble`, `benchmark_serial_vs_parallel`, aliased as `benchmark_scaling`).

It was developed as coursework for MIT **18.337 / 6.7320**, Parallel Computing and Scientific Machine Learning.

License: MIT. See `LICENSE`.

## Status

This is a **research prototype**, not a registered General registry package. It is suitable for reproducible demos, benchmarks, and course reports; SciML-grade polish (compat breadth, broader MTK coverage, etc.) is left as future work.

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

## Interactive Pluto demo

For a walkthrough of the full project state, launch Pluto with the notebook environment:

```bash
julia --project=notebooks -e "using Pkg; Pkg.instantiate()"
julia --project=notebooks -e "using Pluto; Pluto.run()"
```

Then open `notebooks/symbolicmor_current_state.jl`. The notebook covers lifting, snapshots, POD, operator extraction, ROM comparison, scaling helpers, and current limitations.

## Benchmark scripts (`benchmarks/`)

Run from the repo root with `julia --project=. ...`. Suggested order:

| Script | Role |
| --- | --- |
| `benchmarks/cubic_decay_benchmark.jl` | Minimal cubic nonlinearity: verifies lifting and trajectory consistency vs the original ODE. |
| `benchmarks/vanderpol_benchmark.jl` | Mixed cubic term (`x^2*y`-type structure after expansion). |
| `benchmarks/lorenz_benchmark.jl` | Familiar quadratic chaos benchmark; POD + ROM projection spot-check + serial/parallel timing. |
| `benchmarks/allen_cahn_benchmark.jl` | Conservative Allen-Cahn reaction smoke benchmark (`du/dt = -u^3`) plus a documented affine reaction limitation. |
| `benchmarks/quadratic_operator_benchmark.jl` | Dense `H * kron(x, x)` vs tensor-backed quadratic evaluation. |
| `benchmarks/galerkin_scaling_benchmark.jl` | Dense vs tensor Galerkin projection and ROM RHS scaling. |
| `benchmarks/cluster_campaign.jl` | Portable local/HPC benchmark campaign runner with CSV output. |
| `benchmarks/summarize_campaign.jl` | Summarize campaign CSV files into speedup and efficiency plots. |

Parallel scaling smoke test (needs workers, e.g. `-p 4`):

```bash
julia --project=. -p 4 scripts/scaling_benchmark.jl
```

Portable benchmark campaign smoke test:

```bash
julia --project=. -p 2 benchmarks/cluster_campaign.jl --case lorenz --trajectories 16 --tend 1.0 --dt 0.05 --repeats 1 --out benchmark_results/local_p2.csv
julia --project=. benchmarks/summarize_campaign.jl benchmark_results benchmark_results/local_summary
```

## Limitations

- Intended for **polynomial right-hand sides**; broader polynomialization of non-polynomial dynamics remains experimental (`polynomialize.jl`).
- Dense quadratic tensors use **`H * kron(u, u)`** for compatibility; prefer `QuadraticTensor` for larger sparse quadratic systems and ROM RHS evaluation.
- Parallel snapshots use Julia **Distributed**; overhead dominates small ensembles or Windows setups. Linux/cluster runs are cleaner for scaling plots.
- ModelingToolkit support is intentionally narrow: explicit polynomial `ODESystem`s with one `D(x) ~ rhs` equation per state. DAEs, observed-variable elimination, and symbolic parameters without substitution are rejected.

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
