# Cubic decay: minimal example where lifting introduces an auxiliary variable.
#
# Run from repo root:
#   julia --project=. benchmarks/cubic_decay_benchmark.jl

using OrdinaryDiffEq
using LinearAlgebra
using Symbolics
using SymbolicMOR

include(joinpath(@__DIR__, "common.jl"))

println("="^60)
println(" Benchmark: cubic decay (dx/dt = -x^3, dy/dt = -y)")
println("="^60)

@variables x y
rhs = [-x^3, -y]
ls = lift_system([x, y], rhs)
println("\n[1] Lifted system:\n")
println(ls)

A, H, c = extract_operators(ls)
n = length(ls.lifted_vars)

# Original ODE (ground truth for x, y)
function original!(du, u, p, t)
    du[1] = -u[1]^3
    du[2] = -u[2]
    return nothing
end

x0, y0 = 0.8, 0.3
tspan = (0.0, 5.0)

prob_orig = ODEProblem(original!, [x0, y0], tspan)
sol_orig = solve(prob_orig, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.02)

# Lifted IC: evaluate original coordinates, then aux definitions in equation order
vals = Dict{Num, Float64}(x => x0, y => y0)
for eq in ls.aux_eqs
    vals[eq.lhs] = Float64(Symbolics.value(Symbolics.substitute(eq.rhs, vals)))
end
u0_lift = Float64[vals[v] for v in ls.lifted_vars]

prob_lift = ODEProblem(quadratic_state_rhs!, u0_lift, tspan, (A, H, c))
sol_lift = solve(prob_lift, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.02)

max_xy_err = maximum(
    norm(sol_orig.u[i][1:2] - sol_lift.u[i][1:2]) for i in eachindex(sol_orig.t)
)
println("\n[2] Trajectory match (x,y components vs original): max error = ", max_xy_err)
max_xy_err < 1e-5 || @warn "Unexpected drift: check solver tolerances or lifted IC construction."

println("\nDone.")
