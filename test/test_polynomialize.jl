# test/test_polynomialize.jl
#
# Polynomialization correctness for the six supported non-polynomial atoms:
# exp(g), sin(g), cos(g), 1/g, log(g), sqrt(g), tanh(g).
#
# Each case is checked structurally (lift produces a quadratic-ready system)
# and end-to-end (integrating the lifted ODE matches the unlifted trajectory).

using Test
using LinearAlgebra
using Symbolics
using SymbolicMOR
using OrdinaryDiffEq

# Build initial conditions for the lifted state by evaluating each aux
# definition in order, allowing later auxes to depend on earlier ones.
function _lifted_ic(ls::SymbolicMOR.LiftedSystem, u0_orig)
    state = Float64.(collect(u0_orig))
    state_vars = Vector{Num}(collect(ls.original_vars))
    for eq in ls.aux_eqs
        f = Symbolics.build_function(eq.rhs, state_vars; expression=Val{false})
        push!(state, Float64(f(state)))
        push!(state_vars, Num(eq.lhs))
    end
    return state
end

function _make_rhs!(vars, exprs)
    oop, _ = Symbolics.build_function(exprs, vars; expression=Val{false})
    return (du, u, p, t) -> (du .= oop(u); nothing)
end

function _trajectory_error(vars_orig, rhs_orig, u0; tspan, abstol=1e-10, reltol=1e-10, saveat=0.05)
    ls = lift_system(vars_orig, rhs_orig)
    fL! = _make_rhs!(ls.lifted_vars, ls.F)
    fO! = _make_rhs!(vars_orig, rhs_orig)
    u0L = _lifted_ic(ls, u0)
    solL = solve(ODEProblem(fL!, u0L, tspan), Tsit5(); abstol, reltol, saveat)
    solO = solve(ODEProblem(fO!, Float64.(u0), tspan), Tsit5(); abstol, reltol, saveat)
    err = maximum(norm(solL.u[i][1:length(vars_orig)] - solO.u[i]) for i in eachindex(solO.t))
    return ls, err
end

function _max_degree(ls::SymbolicMOR.LiftedSystem)
    isempty(ls.F) && return 0
    SU = Symbolics.SymbolicUtils
    deg = 0
    for f in ls.F
        ex = Symbolics.unwrap(Symbolics.expand(f))
        monomials = SU.isadd(ex) ? collect(SU.arguments(ex)) : [ex]
        for m in monomials
            d = sum(Symbolics.degree(Num(m), v) for v in ls.lifted_vars; init=0)
            deg = max(deg, d)
        end
    end
    return deg
end

@testset "Phase 1 - Polynomialization of non-polynomial atoms" begin
    @testset "exp(g) lifts and integrates" begin
        @variables x
        ls, err = _trajectory_error([x], [exp(-x)], [0.5]; tspan=(0.0, 1.0))
        @test length(ls.aux_eqs) == 1
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "sin(g) and cos(g) close together" begin
        @variables x
        ls, err = _trajectory_error([x], [sin(x)], [0.5]; tspan=(0.0, 2.0))
        @test length(ls.aux_eqs) == 2          # p_sin and p_cos pair
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "1/g lifts and integrates" begin
        @variables x
        ls, err = _trajectory_error([x], [1/(1 + x)], [0.5]; tspan=(0.0, 1.0))
        @test length(ls.aux_eqs) >= 1
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "log(g) lifts and integrates" begin
        @variables x
        ls, err = _trajectory_error([x], [log(2 + x)], [0.5]; tspan=(0.0, 1.0))
        @test length(ls.aux_eqs) >= 2          # p_log and the inv aux it triggers
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "sqrt(g) lifts and integrates" begin
        @variables x
        ls, err = _trajectory_error([x], [sqrt(2 + x)], [0.5]; tspan=(0.0, 1.0))
        @test length(ls.aux_eqs) >= 1
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "tanh(g) closes via the (1 - p^2) identity" begin
        @variables x
        ls, err = _trajectory_error([x], [tanh(x)], [0.5]; tspan=(0.0, 2.0))
        @test length(ls.aux_eqs) >= 1
        @test _max_degree(ls) <= 2
        @test err < 1e-6
    end

    @testset "polynomialize_system returns immediately for already-polynomial RHS" begin
        @variables x y
        result = polynomialize_system([x, y], [x*y, x + y^2])
        @test length(result.aux_defs) == 0
        @test length(result.polynomial_vars) == 2
    end

    @testset "polynomialize_system rejects unsupported atoms" begin
        @variables x
        # asin is not in the supported set
        @test_throws ErrorException polynomialize_system([x], [asin(x)])
    end
end
