# src/lift/mtk_api.jl
#
# Narrow ModelingToolkit integration.
# The core quadratization API remains lift_system(vars, rhs).
# This file adapts explicit ODESystem inputs to the core API.

using ModelingToolkit
using Symbolics

"""
    extract_state_rhs(sys::ModelingToolkit.ODESystem)

Extract `(states, rhs)` from an explicit ModelingToolkit `ODESystem`.

Only systems with one equation of the form `D(x) ~ f(x)` for each state are
supported. DAEs, missing state derivatives, equations for non-state variables,
and RHS expressions containing extra symbolic variables are rejected with an
`ArgumentError`.
"""
function extract_state_rhs(sys::ModelingToolkit.ODESystem)
    states = Num.(ModelingToolkit.get_unknowns(sys))
    equations = ModelingToolkit.get_eqs(sys)
    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)

    rhs_by_state = Dict{Any,Num}()
    expected_lhs = Dict{Any,Any}(Symbolics.unwrap(D(state)) => state for state in states)

    for eq in equations
        lhs = Symbolics.unwrap(eq.lhs)
        if !haskey(expected_lhs, lhs)
            throw(ArgumentError("only explicit state derivative equations D(x) ~ rhs are supported"))
        end

        state = expected_lhs[lhs]
        haskey(rhs_by_state, state) &&
            throw(ArgumentError("duplicate derivative equation for state $state"))

        rhs = Num(eq.rhs)
        _check_rhs_symbols(rhs, states)
        rhs_by_state[state] = rhs
    end

    missing = [state for state in states if !haskey(rhs_by_state, state)]
    isempty(missing) ||
        throw(ArgumentError("missing derivative equations for states: $(join(string.(missing), ", "))"))

    return states, [rhs_by_state[state] for state in states]
end

"""
    lift_system(sys::ModelingToolkit.ODESystem)

Lift an explicit ModelingToolkit `ODESystem` into a quadratic `LiftedSystem`.

This is a convenience adapter for `lift_system(vars, rhs)`. It currently
supports explicit polynomial ODEs whose RHS expressions depend only on state
variables and numeric constants.
"""
function lift_system(sys::ModelingToolkit.ODESystem)
    states, rhs = extract_state_rhs(sys)
    return lift_system(states, rhs)
end

function _check_rhs_symbols(rhs::Num, states::Vector{Num})
    allowed = Set(Symbolics.unwrap.(states))
    extras = Num[]

    for var in Symbolics.get_variables(rhs)
        if !(Symbolics.unwrap(var) in allowed)
            push!(extras, Num(var))
        end
    end

    isempty(extras) ||
        throw(ArgumentError("RHS contains unsupported symbolic variables or parameters: $(join(string.(extras), ", "))"))

    return nothing
end
