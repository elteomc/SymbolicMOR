module SymbolicMOR

# -- Phase 1: Lift --
include("lift/polynomialize.jl")
include("lift/quadratize.jl")

# -- Phase 2: Learn --
include("learn/snapshot.jl")
include("learn/pod_galerkin.jl")

# -- Phase 3: Scale --
include("scale/parallel_snapshot.jl")
include("scale/benchmarks.jl")

export
  # Phase 1
  polynomialize,
  quadratize,
  lift_system,
  # Phase 2
  generate_snapshots,
  compute_pod_basis,
  galerkin_project,
  extract_operators,
  rom_rhs!,
  # Phase 3
  generate_snapshots_parallel,
  benchmark_scaling

end
