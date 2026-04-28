# Van der Pol: mixed cubic nonlinearity (x^2*y term after expansion).
#
# Run from repo root:
#   julia --project=. benchmarks/vanderpol_benchmark.jl

using OrdinaryDiffEq
using LinearAlgebra
using Symbolics
using SymbolicMOR

include(joinpath(@__DIR__, "common.jl"))

const mu = 1.0

println("="^60)
println(" Benchmark: Van der Pol (mu = $mu)")
println("="^60)

@variables x y
rhs = [y, mu * y - mu * x^2 * y - x]
ls = lift_system([x, y], rhs)
println("\n[1] Lifted system summary:")
println("    Original dim: ", length(ls.original_vars))
println("    Lifted dim:   ", length(ls.lifted_vars))
println("    Aux equations: ", length(ls.aux_eqs))

A, H, c = extract_operators(ls)
n = length(ls.lifted_vars)

function original!(du, u, p, t)
    du[1] = u[2]
    du[2] = mu * (1 - u[1]^2) * u[2] - u[1]
    return nothing
end

x0, y0 = 2.0, 0.0
tspan = (0.0, 10.0)

prob_orig = ODEProblem(original!, [x0, y0], tspan)
sol_orig = solve(prob_orig, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

vals = Dict{Num, Float64}(x => x0, y => y0)
for eq in ls.aux_eqs
    vals[eq.lhs] = Float64(Symbolics.value(Symbolics.substitute(eq.rhs, vals)))
end
u0_lift = Float64[vals[v] for v in ls.lifted_vars]

prob_lift = ODEProblem(quadratic_state_rhs!, u0_lift, tspan, (A, H, c))
sol_lift = solve(prob_lift, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

max_xy_err = maximum(
    norm(sol_orig.u[i][1:2] - sol_lift.u[i][1:2]) for i in eachindex(sol_orig.t)
)
println("\n[2] Trajectory match (x,y vs expanded quadratic form): max error = ", max_xy_err)
max_xy_err < 1e-4 || @warn "Large error: stiff VDP may need tighter tolerances or shorter horizon."

println("\nDone.")
