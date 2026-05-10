# scripts/walkthrough_cubic_decay.jl
#
# Narrated walkthrough: cubic decay  dx/dt = -x^3.
# Run with:
#   julia --project=. scripts/walkthrough_cubic_decay.jl
#
# This is the smallest example where lifting actually does something. It is
# a one-dimensional ODE whose right-hand side has degree three, so the
# lifting step has to introduce one auxiliary variable to bring the system
# down to quadratic form. Because the system is so small, every quantity
# (number of aux variables, ROM dimension, trajectory error) is easy to
# inspect by eye.

using Symbolics
using SymbolicMOR
using OrdinaryDiffEq
using LinearAlgebra
using Random

# ---------------------------------------------------------------------
# 1. The original problem
# ---------------------------------------------------------------------
#
# We pick the simplest scalar ODE that is not quadratic:
#
#       dx/dt = -x^3
#
# The cube on the right makes this not directly amenable to Galerkin
# projection of the form A a + H (a kron a) + c. Our lift step rewrites
# it into a quadratic system in an enlarged state.

println("="^70)
println(" Cubic decay walkthrough:  dx/dt = -x^3 ")
println("="^70)

@variables x
ls = lift_system([x], [-x^3])

println("\nOriginal state dim: ", length(ls.original_vars))
println("Lifted state dim:   ", length(ls.lifted_vars))
println("Aux equations introduced:")
for eq in ls.aux_eqs
    println("  ", eq.lhs, " = ", eq.rhs)
end
println("Lifted RHS:")
for (v, f) in zip(ls.lifted_vars, ls.F)
    println("  d", v, "/dt = ", f)
end

# ---------------------------------------------------------------------
# 2. What just happened
# ---------------------------------------------------------------------
#
# The quadratization step factors the cubic monomial x^3 into x * x and
# introduces w1 = x*x. The lifted dynamics are:
#
#       dx/dt  = -x * w1
#       dw1/dt = 2 x * (dx/dt) = -2 x^2 * w1 = -2 w1 * w1   = -2 w1^2
#
# Both right-hand sides are quadratic in (x, w1). The dedup step in
# quadratize.jl is what keeps -2 w1*w1 from spawning *another* aux.

# ---------------------------------------------------------------------
# 3. Snapshots
# ---------------------------------------------------------------------
#
# Build a callable lifted vector field and integrate from a small ensemble
# of initial conditions. With this much structure (one degree of freedom),
# 12 trajectories give plenty of data for a meaningful SVD.

G! = build_lifted_rhs(ls)

function lifted_ic(ls, u0_orig)
    state = Float64.(collect(u0_orig))
    state_vars = Vector{Num}(collect(ls.original_vars))
    for eq in ls.aux_eqs
        f = Symbolics.build_function(eq.rhs, state_vars; expression=Val{false})
        push!(state, Float64(f(state)))
        push!(state_vars, Num(eq.lhs))
    end
    return state
end

Random.seed!(0)
u0_lift_set = [lifted_ic(ls, [v]) for v in 1.5 .* (rand(12) .- 0.5)]
X = generate_snapshots(G!, u0_lift_set, (0.0, 4.0); dt=0.02)
println("\nSnapshot matrix size: ", size(X))

# ---------------------------------------------------------------------
# 4. POD basis
# ---------------------------------------------------------------------
#
# Even with two state variables, POD finds that essentially all the
# trajectory energy lives along one direction in the lifted state space:
# w1 is determined by x along solutions, so the snapshots concentrate
# on the curve {(x, x^2) : x in R}. Keep r = 1 to see the contrast
# with r = 2 (which is exact up to solver tolerance).

V, sigma, energy = compute_pod_basis(X, 1)
println("\nSingular values: ", round.(sigma; sigdigits=4))
println("Energy retained by r = 1 mode: ", round(energy * 100; digits=4), "%")

# ---------------------------------------------------------------------
# 5. Build both ROM forms
# ---------------------------------------------------------------------

A, Q, c = extract_quadratic_tensor(ls)
A_hat, Q_hat, c_hat = galerkin_project(A, Q, c, V)
rom_simple! = build_simple_rom_rhs(ls, V)

# ---------------------------------------------------------------------
# 6. Compare ROM trajectory to full lifted trajectory
# ---------------------------------------------------------------------
#
# The reduced state is one-dimensional. We project the test IC into
# the basis V, integrate the ROM, then map back to the lifted state to
# compare the x component to the full simulation.

u0_full = lifted_ic(ls, [0.8])
a0 = V' * u0_full
tspan = (0.0, 4.0)

sol_full     = solve(ODEProblem(G!, u0_full, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_exp  = solve(ODEProblem(rom_rhs!, a0, tspan, (A_hat, Q_hat, c_hat)), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_simp = solve(ODEProblem(rom_simple!, a0, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

err_form = maximum(norm(sol_rom_exp.u[i] - sol_rom_simp.u[i]) for i in eachindex(sol_rom_exp.t))
err_x    = maximum(abs((V * sol_rom_exp.u[i])[1] - sol_full.u[i][1]) for i in eachindex(sol_full.t))

println("\nSimple vs explicit ROM (should be solver tol): ", err_form)
println("Reduced (r=1) vs full lifted, x component:     ", err_x)

# ---------------------------------------------------------------------
# 7. Closing observation
# ---------------------------------------------------------------------
#
# r = 1 captures the dominant direction (x decays toward zero, and so does
# w1 = x^2 along solutions). The residual error reflects that one mode
# cannot represent the curve {(x, x^2)} exactly: there is information in
# the second mode too. Bumping r to 2 makes the ROM exact up to the
# integrator's tolerance.

println("\nDone.")
