module SymbolicMOR

# -- Phase 1: Lift --
include("lift/polynomialize.jl")
include("lift/quadratize.jl")
include("lift/mtk_api.jl")

# -- Phase 2: Learn --
include("learn/snapshot.jl")
include("learn/quadratic_operator.jl")
include("learn/pod_galerkin.jl")

# -- Phase 3: Scale --
include("scale/parallel_snapshot.jl")
include("scale/ensemble_snapshot.jl")
include("scale/benchmarks.jl")

export
  # Phase 1
  polynomialize_system,
  PolynomializationResult,
  quadratize,
  lift_system,
  extract_state_rhs,
  # Phase 2
  generate_snapshots,
  QuadraticTensor,
  evaluate_quadratic!,
  dense_matrix,
  sparse_matrix,
  quadratic_rhs!,
  compute_pod_basis,
  galerkin_project,
  extract_operators_dense,
  extract_operators_sparse,
  extract_quadratic_tensor,
  extract_operators,
  rom_rhs!,
  # Phase 3
  generate_snapshots_parallel,
  generate_snapshots_ensemble,
  benchmark_scaling,
  benchmark_serial_vs_parallel
end
