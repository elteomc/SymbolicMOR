# src/learn/quadratic_operator.jl
#
# Sparse/tensor representation for quadratic operators. A QuadraticTensor stores
# only nonzero quadratic terms and evaluates them without forming kron(x, x).

using LinearAlgebra
using SparseArrays

"""
    QuadraticTensor(n_out, n_state, rows, cols1, cols2, vals)

Coordinate representation of a quadratic map.

Each stored entry contributes

    out[rows[k]] += vals[k] * x[cols1[k]] * x[cols2[k]]

Pairs are stored canonically with `cols1[k] <= cols2[k]`, and duplicate
`(row, col1, col2)` entries are coalesced at construction time.
"""
struct QuadraticTensor{T}
    n_out::Int
    n_state::Int
    rows::Vector{Int}
    cols1::Vector{Int}
    cols2::Vector{Int}
    vals::Vector{T}

    function QuadraticTensor(
        n_out::Integer,
        n_state::Integer,
        rows::AbstractVector{<:Integer},
        cols1::AbstractVector{<:Integer},
        cols2::AbstractVector{<:Integer},
        vals::AbstractVector{T}
    ) where {T}
        lengths = (length(rows), length(cols1), length(cols2), length(vals))
        all(==(first(lengths)), lengths) ||
            throw(ArgumentError("rows, cols1, cols2, and vals must have equal length"))

        accum = Dict{Tuple{Int,Int,Int}, T}()
        for k in eachindex(vals)
            row = Int(rows[k])
            c1 = Int(cols1[k])
            c2 = Int(cols2[k])

            1 <= row <= n_out || throw(ArgumentError("row index out of bounds"))
            1 <= c1 <= n_state || throw(ArgumentError("cols1 index out of bounds"))
            1 <= c2 <= n_state || throw(ArgumentError("cols2 index out of bounds"))

            if c2 < c1
                c1, c2 = c2, c1
            end

            key = (row, c1, c2)
            accum[key] = get(accum, key, zero(T)) + vals[k]
        end

        out_rows = Int[]
        out_cols1 = Int[]
        out_cols2 = Int[]
        out_vals = T[]

        for (key, val) in sort!(collect(accum); by = first)
            iszero(val) && continue
            row, c1, c2 = key
            push!(out_rows, row)
            push!(out_cols1, c1)
            push!(out_cols2, c2)
            push!(out_vals, val)
        end

        return new{T}(Int(n_out), Int(n_state), out_rows, out_cols1, out_cols2, out_vals)
    end
end

Base.length(Q::QuadraticTensor) = length(Q.vals)

"""
    evaluate_quadratic!(out, Q, x; reset=true)

Evaluate `Q(x, x)` in-place. When `reset=true`, `out` is zeroed before
accumulating quadratic terms.
"""
function evaluate_quadratic!(out::AbstractVector, Q::QuadraticTensor, x::AbstractVector; reset::Bool = true)
    length(out) == Q.n_out || throw(DimensionMismatch("out length must equal Q.n_out"))
    length(x) == Q.n_state || throw(DimensionMismatch("x length must equal Q.n_state"))

    reset && fill!(out, zero(eltype(out)))
    for k in eachindex(Q.vals)
        out[Q.rows[k]] += Q.vals[k] * x[Q.cols1[k]] * x[Q.cols2[k]]
    end
    return out
end

"""
    dense_matrix(Q)

Convert `Q` to the dense matrix layout `H` used by
`H * kron(x, x)`. Mixed terms are split symmetrically across `(p, q)` and
`(q, p)` columns.
"""
function dense_matrix(Q::QuadraticTensor{T}) where {T}
    H = zeros(T, Q.n_out, Q.n_state^2)
    for k in eachindex(Q.vals)
        row = Q.rows[k]
        p = Q.cols1[k]
        q = Q.cols2[k]
        val = Q.vals[k]

        if p == q
            H[row, (p - 1) * Q.n_state + q] += val
        else
            half_val = val / 2
            H[row, (p - 1) * Q.n_state + q] += half_val
            H[row, (q - 1) * Q.n_state + p] += half_val
        end
    end
    return H
end

"""
    sparse_matrix(Q)

Convert `Q` to a sparse matrix in the current `n_out x n_state^2` layout.
"""
function sparse_matrix(Q::QuadraticTensor{T}) where {T}
    rows = Int[]
    cols = Int[]
    vals = T[]

    for k in eachindex(Q.vals)
        row = Q.rows[k]
        p = Q.cols1[k]
        q = Q.cols2[k]
        val = Q.vals[k]

        if p == q
            push!(rows, row)
            push!(cols, (p - 1) * Q.n_state + q)
            push!(vals, val)
        else
            half_val = val / 2
            push!(rows, row)
            push!(cols, (p - 1) * Q.n_state + q)
            push!(vals, half_val)
            push!(rows, row)
            push!(cols, (q - 1) * Q.n_state + p)
            push!(vals, half_val)
        end
    end

    return sparse(rows, cols, vals, Q.n_out, Q.n_state^2)
end

"""
    quadratic_rhs!(du, u, (A, Q, c), t)

In-place RHS for a quadratic system where the quadratic operator is a
`QuadraticTensor`.
"""
function quadratic_rhs!(du, u, params, t)
    A, Q, c = params
    du .= A * u .+ c
    evaluate_quadratic!(du, Q, u; reset = false)
    return nothing
end
