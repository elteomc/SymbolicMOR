# src/scale/parallel_snapshot.jl
#
# Phase 3 - Parallelized Snapshot Generation
#
# Since each trajectory is independent, we can distribute ODE solves
# across available CPU cores using Julia's Distributed standard library.

using Distributed
using OrdinaryDiffEq
using LinearAlgebra

"""
    generate_snapshots_parallel(f!, u0_ensemble, tspan; kwargs...)

Parallel version of generate_snapshots. Distributes ODE solves across
all available workers using @distributed.

# Usage
Before calling this, ensure workers are added:
    using Distributed
    addprocs(4)
    @everywhere using LiftAndLearnMOR

Then:
    X = generate_snapshots_parallel(lorenz!, u0s, (0.0, 10.0))

The speedup should scale approximately linearly with the number of workers
for large ensembles (communication overhead is minimal since each job
returns a matrix and trajectories are independent).
"""
function generate_snapshots_parallel(
  f!,
  u0_ensemble::Vector{<:AbstractVector},
  tspan::Tuple{<:Real,<:Real};
  dt::Float64=0.01,
  solver=Tsit5(),
  kwargs...
)
  n_workers = nworkers()

  if n_workers == 1
    @warn "Only 1 worker available. Running serial version. Use addprocs(N) to add workers."
    return generate_snapshots(f!, u0_ensemble, tspan; dt, solver, kwargs...)
  end

  snapshot_blocks = @distributed (hcat_reducer) for u0 in u0_ensemble
    prob = ODEProblem(f!, u0, tspan)
    sol = solve(prob, solver; saveat=dt, kwargs...)
    hcat(sol.u...)
  end

  return snapshot_blocks
end

# Reduction function: horizontally concatenate matrices returned by workers
hcat_reducer(A, B) = hcat(A, B)
