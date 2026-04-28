# Allen-Cahn reaction benchmark.
#
# Run from repo root:
#   julia --project=. benchmarks/allen_cahn_benchmark.jl
#
# This intentionally uses the conservative pure-cubic reaction du/dt = -u^3. A fuller
# Allen-Cahn reaction term, du/dt = u - u^3, currently exposes a symbolic simplification
# bottleneck in lift_system and is kept out of this default benchmark so the script is
# robust and finishes quickly enough for normal use.

using OrdinaryDiffEq
using LinearAlgebra
using Symbolics
using SymbolicMOR

include(joinpath(@__DIR__, "common.jl"))

println("="^60)
println(" Benchmark: Allen-Cahn reaction smoke test (du/dt = -u^3)")
println("="^60)

@variables u
rhs = [-u^3]
ls = lift_system([u], rhs)

println("\n[1] Lifted system:")
println("    Original dim: ", length(ls.original_vars))
println("    Lifted dim:   ", length(ls.lifted_vars))
println("    Aux equations: ", length(ls.aux_eqs))

A, H, c = extract_operators(ls)

function original!(du, state, p, t)
    du[1] = -state[1]^3
    return nothing
end

u0 = 0.8
tspan = (0.0, 5.0)
saveat = 0.02

prob_orig = ODEProblem(original!, [u0], tspan)
sol_orig = solve(prob_orig, Tsit5(); abstol=1e-10, reltol=1e-10, saveat)

vals = Dict{Num, Float64}(u => u0)
for eq in ls.aux_eqs
    vals[eq.lhs] = Float64(Symbolics.value(Symbolics.substitute(eq.rhs, vals)))
end
u0_lift = Float64[vals[v] for v in ls.lifted_vars]

prob_lift = ODEProblem(quadratic_state_rhs!, u0_lift, tspan, (A, H, c))
sol_lift = solve(prob_lift, Tsit5(); abstol=1e-10, reltol=1e-10, saveat)

max_err = maximum(abs(sol_orig.u[i][1] - sol_lift.u[i][1]) for i in eachindex(sol_orig.t))
println("\n[2] Trajectory match (u component vs original): max error = ", max_err)
max_err < 1e-5 || @warn "Unexpected drift: check solver tolerances or lifted IC construction."

println("\n[3] Note: affine Allen-Cahn reaction u - u^3 is currently documented as a limitation.")
println("\nDone.")
