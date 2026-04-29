# Compare dense H*kron(x,x) evaluation against QuadraticTensor evaluation.
#
# Run from repo root:
#   julia --project=. benchmarks/quadratic_operator_benchmark.jl

using BenchmarkTools
using LinearAlgebra
using Random
using SymbolicMOR

println("="^60)
println(" Benchmark: dense vs tensor quadratic operator evaluation")
println("="^60)

Random.seed!(123)
n = 80
n_terms = 4n

rows = rand(1:n, n_terms)
cols1 = rand(1:n, n_terms)
cols2 = rand(1:n, n_terms)
vals = randn(n_terms)

Q = QuadraticTensor(n, n, rows, cols1, cols2, vals)
H = dense_matrix(Q)
x = randn(n)
out = zeros(n)

dense_eval() = H * kron(x, x)
tensor_eval!() = evaluate_quadratic!(out, Q, x)

dense_result = dense_eval()
tensor_result = tensor_eval!()
@assert isapprox(tensor_result, dense_result; atol=1e-10, rtol=1e-10)

println("\n[1] Operator dimensions:")
println("    n = ", n)
println("    stored tensor terms = ", length(Q))
println("    dense H size = ", size(H))

println("\n[2] Dense H*kron timing:")
display(@benchmark dense_eval())

println("\n[3] Tensor evaluation timing:")
display(@benchmark tensor_eval!())

println("\nDone.")
