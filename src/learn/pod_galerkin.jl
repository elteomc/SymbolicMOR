# src/learn/pod_galerkin.jl
#
# Phase 2 - POD + Galerkin Projection
#
# Given the snapshot matrix X, compute a low-dimensional basis via
# Proper Orthogonal Decomposition (POD), then project the lifted
# quadratic ODE onto this basis to obtain the Reduced-Order Model (ROM).

using LinearAlgebra
using SparseArrays

"""
    compute_pod_basis(X, r; energy_threshold=0.9999)

Compute a POD basis from the snapshot matrix X in R^(n x K).

# Arguments
- X                : snapshot matrix (rows = state dims, cols = snapshots)
- r                : number of POD modes to retain (pass `nothing` to use threshold)
- energy_threshold : if r is `nothing`, retain enough modes to capture this
                     fraction of total energy (default: 99.99%)

# Returns
(Phi, sigma, energy) where:
  - Phi::Matrix{Float64}   : POD basis, shape (n, r), orthonormal columns
  - sigma::Vector{Float64} : singular values
  - energy::Float64        : fraction of energy captured by chosen modes
"""
function compute_pod_basis(X::Matrix{Float64}, r::Union{Int,Nothing}=nothing;
                           energy_threshold::Float64=0.9999)
    U, sigma, _ = svd(X)

    # Determine r from energy threshold if not specified
    if isnothing(r)
        total_energy = sum(sigma.^2)
        cumulative   = cumsum(sigma.^2) ./ total_energy
        r = findfirst(>=(energy_threshold), cumulative)
        isnothing(r) && (r = length(sigma))
        @info "POD: retaining r=$r modes ($(round(cumulative[r]*100, digits=4))% energy)"
    end

    r = min(r, size(U, 2))
    Phi = U[:, 1:r]
    energy = sum(sigma[1:r].^2) / sum(sigma.^2)

    return Phi, sigma, energy
end

"""
    galerkin_project(A, H, c, Phi)

Project the lifted quadratic RHS onto the POD basis Phi to produce the ROM.

For a lifted system  s_dot = F(s)  (quadratic), let  s approx Phi a,
then the ROM is:
    a_dot = Phi' * F(Phi * a)

In matrix form for a quadratic system
    F(s) = A*s + H*(s kron s) + c

the ROM operators are:
    A_hat = Phi' * A * Phi
    H_hat = Phi' * H * (Phi kron Phi)
    c_hat = Phi' * c

# Arguments
- A   : linear operator matrix (n x n)
- H   : quadratic operator (n x n^2)
- c   : constant term (n,)
- Phi : POD basis (n x r)

# Returns
(A_hat, H_hat, c_hat) for the ROM ODE:
    a_dot = A_hat*a + H_hat*(a kron a) + c_hat
"""
function galerkin_project(A::AbstractMatrix, H::AbstractMatrix,
                          c::AbstractVector, Phi::AbstractMatrix)
    r = size(Phi, 2)
    A_hat = Phi' * A * Phi
    H_hat = Phi' * H * kron(Phi, Phi)
    c_hat = Phi' * c
    return A_hat, H_hat, c_hat
end

"""
    galerkin_project(A, Q::QuadraticTensor, c, Phi)

Project a quadratic tensor operator without forming `kron(Phi, Phi)`.
Returns `(A_hat, Q_hat, c_hat)`, where `Q_hat` is also a `QuadraticTensor`.
"""
function galerkin_project(A::AbstractMatrix, Q::QuadraticTensor,
                          c::AbstractVector, Phi::AbstractMatrix)
    r = size(Phi, 2)
    A_hat = Phi' * A * Phi
    c_hat = Phi' * c

    T = promote_type(eltype(Phi), eltype(Q.vals), eltype(A), eltype(c))
    reduced = zeros(T, r, r, r)

    for k in eachindex(Q.vals)
        i = Q.rows[k]
        p = Q.cols1[k]
        q = Q.cols2[k]
        val = Q.vals[k]

        for alpha in 1:r
            row_coeff = Phi[i, alpha] * val
            iszero(row_coeff) && continue

            for beta in 1:r, gamma in 1:r
                coeff = row_coeff * Phi[p, beta] * Phi[q, gamma]
                iszero(coeff) && continue

                col1 = min(beta, gamma)
                col2 = max(beta, gamma)
                reduced[alpha, col1, col2] += coeff
            end
        end
    end

    rows = Int[]
    cols1 = Int[]
    cols2 = Int[]
    vals = T[]

    for alpha in 1:r, beta in 1:r, gamma in beta:r
        coeff = reduced[alpha, beta, gamma]
        iszero(coeff) && continue

        push!(rows, alpha)
        push!(cols1, beta)
        push!(cols2, gamma)
        push!(vals, coeff)
    end

    Q_hat = QuadraticTensor(r, r, rows, cols1, cols2, vals)
    return A_hat, Q_hat, c_hat
end

"""
    build_lifted_rhs(ls::LiftedSystem)

Compile the lifted vector field `G(s) = ls.F(s)` into a callable
`(du, u, p, t) -> nothing` suitable for `ODEProblem`. Useful as the
full-order RHS, and as a building block for the simple Galerkin ROM
which evaluates G inside the reduced-state ODE.
"""
function build_lifted_rhs(ls::LiftedSystem)
    oop, _ = Symbolics.build_function(ls.F, ls.lifted_vars; expression=Val{false})
    return (du, u, p, t) -> (du .= oop(u); nothing)
end

"""
    build_simple_rom_rhs(ls::LiftedSystem, V::AbstractMatrix; z_mean=zeros(size(V,1)))

Build a closure for the *simple* (function-evaluating) Galerkin ROM:

```math
\\dot{a} = V^\\top \\, G(\\bar{z} + V a)
```

where `G` is the full lifted vector field and `V` is the POD basis (`N \\times r`),
`z_mean` is the optional mean offset. Returns a callable
`(da, a, p, t) -> nothing` suitable for `ODEProblem`. Internal buffers are
preallocated, so each call is allocation-free.

This is the pedagogically simplest form of intrusive MOR. For an
allocation-free *explicit-operator* ROM that avoids evaluating G during
reduced simulation, use `extract_operators` + `galerkin_project` + `rom_rhs!`.
"""
function build_simple_rom_rhs(ls::LiftedSystem, V::AbstractMatrix;
                              z_mean::AbstractVector = zeros(size(V, 1)))
    size(V, 1) == length(z_mean) ||
        throw(DimensionMismatch("V has $(size(V, 1)) rows but z_mean has $(length(z_mean)) entries"))
    size(V, 1) == length(ls.lifted_vars) ||
        throw(DimensionMismatch("V has $(size(V, 1)) rows but ls has $(length(ls.lifted_vars)) lifted vars"))

    G! = build_lifted_rhs(ls)
    N = size(V, 1)
    s_buf  = zeros(N)
    Gs_buf = zeros(N)
    z_local = collect(Float64, z_mean)
    Vt = transpose(V)

    return function (da, a, p, t)
        mul!(s_buf, V, a)
        s_buf .+= z_local
        G!(Gs_buf, s_buf, p, t)
        mul!(da, Vt, Gs_buf)
        return nothing
    end
end

function _apply_linear_constant!(out, A, x, c)
    mul!(out, A, x)
    out .+= c
    return out
end

function _add_dense_quadratic!(out, H, x)
    n = length(x)
    for row in axes(H, 1)
        acc = zero(eltype(out))
        for col1 in 1:n, col2 in 1:n
            coeff = H[row, (col1 - 1) * n + col2]
            iszero(coeff) && continue
            acc += coeff * x[col1] * x[col2]
        end
        out[row] += acc
    end
    return out
end

"""
    rom_rhs!(da, a, (A_hat, H_hat, c_hat), t)

In-place RHS for the reduced-order model. Pass to `ODEProblem`.

    a_dot = A_hat*a + H_hat*(a kron a) + c_hat
"""
function rom_rhs!(da, a, params, t)
    A_hat, H_hat, c_hat = params
    _apply_linear_constant!(da, A_hat, a, c_hat)
    if H_hat isa QuadraticTensor
        evaluate_quadratic!(da, H_hat, a; reset = false)
    else
        _add_dense_quadratic!(da, H_hat, a)
    end
    return nothing
end

function _extract_linear_constant(ls::LiftedSystem)
    n    = length(ls.lifted_vars)
    s    = ls.lifted_vars
    F    = ls.F
    sub0 = Dict(v => 0.0 for v in s)

    _eval(expr) = Float64(Symbolics.value(Symbolics.substitute(expr, sub0)))

    # constant term
    c = [_eval(F[i]) for i in 1:n]

    # linear term
    A = zeros(n, n)
    for i in 1:n, j in 1:n
        A[i,j] = _eval(Symbolics.derivative(F[i], s[j]))
    end

    return A, c, s, F, _eval
end

"""
    extract_quadratic_tensor(ls::LiftedSystem)

Extract `(A, Q, c)` from a `LiftedSystem`, where `Q` is a sparse coordinate
tensor representation of the quadratic terms.
"""
function extract_quadratic_tensor(ls::LiftedSystem)
    A, c, s, F, _eval = _extract_linear_constant(ls)
    n = length(s)

    rows = Int[]
    cols1 = Int[]
    cols2 = Int[]
    vals = Float64[]

    for i in 1:n, p in 1:n, q in p:n
        d2 = Symbolics.derivative(Symbolics.derivative(F[i], s[p]), s[q])
        d2_val = _eval(d2)
        coeff = p == q ? d2_val / 2 : d2_val
        iszero(coeff) && continue

        push!(rows, i)
        push!(cols1, p)
        push!(cols2, q)
        push!(vals, coeff)
    end

    return A, QuadraticTensor(n, n, rows, cols1, cols2, vals), c
end

"""
    extract_operators_sparse(ls::LiftedSystem)

Extract `(A, H, c)` where `H` is a sparse matrix in the current
`n x n^2` quadratic operator layout.
"""
function extract_operators_sparse(ls::LiftedSystem)
    A, Q, c = extract_quadratic_tensor(ls)
    return sparse(A), sparse_matrix(Q), c
end

"""
    extract_operators_dense(ls::LiftedSystem)

Extract dense numerical operators `(A, H, c)` from a `LiftedSystem`.
This is the compatibility path used by `extract_operators`.
"""
function extract_operators_dense(ls::LiftedSystem)
    A, Q, c = extract_quadratic_tensor(ls)
    return A, dense_matrix(Q), c
end

"""
    extract_operators(ls::LiftedSystem)

Extract the dense numerical matrices A, H, c from a LiftedSystem such that:
    F(s) = A*s + H*(kron(s,s)) + c

This preserves the original dense API. For larger systems, prefer
`extract_quadratic_tensor` or `extract_operators_sparse`.
"""
function extract_operators(ls::LiftedSystem)
    return extract_operators_dense(ls)
end
