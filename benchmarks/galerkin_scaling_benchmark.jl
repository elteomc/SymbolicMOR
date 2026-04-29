# Compare dense and tensor Galerkin projection paths plus ROM RHS evaluation.
#
# Run from repo root:
#   julia --project=. benchmarks/galerkin_scaling_benchmark.jl

using BenchmarkTools
using LinearAlgebra
using Random
using SymbolicMOR

println("="^60)
println(" Benchmark: Galerkin projection and ROM RHS scaling")
println("="^60)

Random.seed!(321)

function random_tensor_operator(n::Int, n_terms::Int)
    return QuadraticTensor(
        n,
        n,
        rand(1:n, n_terms),
        rand(1:n, n_terms),
        rand(1:n, n_terms),
        randn(n_terms),
    )
end

function old_tensor_project(A, Q::QuadraticTensor, c, Phi)
    r = size(Phi, 2)
    A_hat = Phi' * A * Phi
    c_hat = Phi' * c

    rows = Int[]
    cols1 = Int[]
    cols2 = Int[]
    vals = Float64[]

    for k in eachindex(Q.vals)
        i = Q.rows[k]
        p = Q.cols1[k]
        q = Q.cols2[k]
        val = Q.vals[k]

        for alpha in 1:r, beta in 1:r, gamma in 1:r
            coeff = Phi[i, alpha] * val * Phi[p, beta] * Phi[q, gamma]
            iszero(coeff) && continue

            push!(rows, alpha)
            push!(cols1, beta)
            push!(cols2, gamma)
            push!(vals, coeff)
        end
    end

    return A_hat, QuadraticTensor(r, r, rows, cols1, cols2, vals), c_hat
end

function build_case(n::Int, r::Int, n_terms::Int)
    A = randn(n, n) ./ sqrt(n)
    Q = random_tensor_operator(n, n_terms)
    H = dense_matrix(Q)
    c = randn(n)
    Phi = Matrix(qr(randn(n, r)).Q)[:, 1:r]

    A_dense, H_dense, c_dense = galerkin_project(A, H, c, Phi)
    A_tensor, Q_tensor, c_tensor = galerkin_project(A, Q, c, Phi)

    @assert isapprox(A_tensor, A_dense; atol=1e-10, rtol=1e-10)
    @assert isapprox(c_tensor, c_dense; atol=1e-10, rtol=1e-10)
    @assert isapprox(dense_matrix(Q_tensor), H_dense; atol=1e-10, rtol=1e-10)

    a = randn(r)
    da_dense = zeros(r)
    da_tensor = zeros(r)

    rom_rhs!(da_dense, a, (A_dense, H_dense, c_dense), 0.0)
    rom_rhs!(da_tensor, a, (A_tensor, Q_tensor, c_tensor), 0.0)
    @assert isapprox(da_tensor, da_dense; atol=1e-10, rtol=1e-10)

    return (; A, Q, H, c, Phi, A_dense, H_dense, c_dense, A_tensor, Q_tensor, c_tensor, a, da_dense, da_tensor)
end

cases = [
    (n = 40, r = 4, n_terms = 160),
    (n = 80, r = 6, n_terms = 320),
    (n = 120, r = 8, n_terms = 480),
]

for case in cases
    println("\nCase: n=$(case.n), r=$(case.r), tensor terms=$(case.n_terms)")
    data = build_case(case.n, case.r, case.n_terms)
    A = data.A
    Q = data.Q
    H = data.H
    c = data.c
    Phi = data.Phi
    A_dense = data.A_dense
    H_dense = data.H_dense
    c_dense = data.c_dense
    A_tensor = data.A_tensor
    Q_tensor = data.Q_tensor
    c_tensor = data.c_tensor
    a = data.a
    da_dense = data.da_dense
    da_tensor = data.da_tensor

    println("  Dense projection:")
    display(@benchmark galerkin_project($A, $H, $c, $Phi))

    println("  Old tensor projection (push/coalesce baseline):")
    display(@benchmark old_tensor_project($A, $Q, $c, $Phi))

    println("  Optimized tensor projection:")
    display(@benchmark galerkin_project($A, $Q, $c, $Phi))

    println("  Dense ROM RHS:")
    display(@benchmark rom_rhs!($da_dense, $a, (($A_dense, $H_dense, $c_dense)), 0.0))

    println("  Tensor ROM RHS:")
    display(@benchmark rom_rhs!($da_tensor, $a, (($A_tensor, $Q_tensor, $c_tensor)), 0.0))
end

println("\nDone.")
