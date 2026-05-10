# API

## Lifting

```@docs
SymbolicMOR.LiftedSystem
polynomialize
lift_system
extract_state_rhs
quadratize
```

## Learning

```@docs
generate_snapshots
QuadraticTensor
evaluate_quadratic!
dense_matrix
sparse_matrix
quadratic_rhs!
compute_pod_basis
galerkin_project
extract_operators_dense
extract_operators_sparse
extract_quadratic_tensor
extract_operators
rom_rhs!
```

## Scaling

```@docs
generate_snapshots_parallel
generate_snapshots_ensemble
benchmark_serial_vs_parallel
benchmark_scaling
```
