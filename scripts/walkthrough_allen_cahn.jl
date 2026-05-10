# scripts/walkthrough_allen_cahn.jl
#
# Narrated walkthrough: Allen-Cahn-style cubic reaction.
# Run with:
#   julia --project=. scripts/walkthrough_allen_cahn.jl
#
# This is the smallest interesting reaction term from the Allen-Cahn
# family: du/dt = -u^3. We use the scalar version here to keep the
# walkthrough quick and the math close to the slides; a 1D mesh
# generalization is sketched at the end. The cubic nonlinearity is the
# part that makes lifting necessary, and its closure is identical to
# the cubic_decay walkthrough's: w1 = u * u, dw1/dt = -2 w1^2.

using Symbolics
using SymbolicMOR
using OrdinaryDiffEq
using LinearAlgebra
using Random

println("="^70)
println(" Allen-Cahn walkthrough (scalar):  du/dt = -u^3 ")
println("="^70)

# ---------------------------------------------------------------------
# 1. The system
# ---------------------------------------------------------------------
#
# In the full Allen-Cahn PDE
#       u_t = epsilon * u_xx + u - u^3,
# the reaction term u - u^3 is the source of nonlinearity. With the
# diffusion piece epsilon * u_xx discretized linearly (a tridiagonal
# matrix), the nonlinear lift challenge collapses to handling u - u^3
# at each grid point. We focus on the reaction here. (The bilinear
# u term currently exposes a Symbolics simplification slowdown when
# combined with -u^3; the benchmark folder uses pure -u^3 to dodge
# that. The lift logic itself is the same.)

@variables u
ls = lift_system([u], [-u^3])

println("\nOriginal state dim: ", length(ls.original_vars))
println("Lifted state dim:   ", length(ls.lifted_vars))
println("Aux equations introduced:")
for eq in ls.aux_eqs
    println("  ", eq.lhs, " = ", eq.rhs)
end

# ---------------------------------------------------------------------
# 2. What lift produced
# ---------------------------------------------------------------------
#
# Same closure as cubic decay: one aux w1 = u^2, with
#       du/dt  = -u * w1
#       dw1/dt = -2 * w1 * w1
# Both quadratic in (u, w1). The ROM extracted from this is what every
# grid point in a discretized Allen-Cahn would see, multiplied across
# space.

# ---------------------------------------------------------------------
# 3. Snapshots
# ---------------------------------------------------------------------
#
# In a real PDE setting the snapshots would be high-dimensional state
# vectors. Here we sample a one-dimensional slice. The dynamics drive u
# toward zero (the unstable equilibrium of the full Allen-Cahn but the
# only equilibrium with the reaction reduced to -u^3), so the snapshot
# matrix is dominated by the relaxation curve.

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

Random.seed!(11)
u0_lift_set = [lifted_ic(ls, [v]) for v in 0.2 .+ 0.6 .* (rand(20) .- 0.5)]
X = generate_snapshots(G!, u0_lift_set, (0.0, 6.0); dt=0.02)
println("\nSnapshot matrix size: ", size(X))

# ---------------------------------------------------------------------
# 4. POD basis
# ---------------------------------------------------------------------
#
# The relaxation u(t) -> 0 has essentially one direction in the
# (u, w1) lifted space. We expect r = 1 to capture nearly all the
# energy and r = 2 to be exact up to solver tolerance.

r = 1
V, sigma, energy = compute_pod_basis(X, r)
println("\nSingular values: ", round.(sigma; sigdigits=4))
println("Energy retained by r = $r mode: ", round(energy * 100; digits=4), "%")

# ---------------------------------------------------------------------
# 5. Both ROM forms
# ---------------------------------------------------------------------

A, Q, c = extract_quadratic_tensor(ls)
A_hat, Q_hat, c_hat = galerkin_project(A, Q, c, V)
rom_simple! = build_simple_rom_rhs(ls, V)

# ---------------------------------------------------------------------
# 6. Compare against the full simulation
# ---------------------------------------------------------------------

u0_full = lifted_ic(ls, [0.5])
a0 = V' * u0_full
tspan = (0.0, 6.0)

sol_full     = solve(ODEProblem(G!, u0_full, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_exp  = solve(ODEProblem(rom_rhs!, a0, tspan, (A_hat, Q_hat, c_hat)), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_simp = solve(ODEProblem(rom_simple!, a0, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

err_form = maximum(norm(sol_rom_exp.u[i] - sol_rom_simp.u[i]) for i in eachindex(sol_rom_exp.t))
err_u    = maximum(abs((V * sol_rom_exp.u[i])[1] - sol_full.u[i][1]) for i in eachindex(sol_full.t))

println("\nSimple vs explicit ROM (should be solver tol): ", err_form)
println("Reduced (r=$r) vs full lifted, u component:    ", err_u)

# ---------------------------------------------------------------------
# 7. Toward a 1D mesh
# ---------------------------------------------------------------------
#
# To go from this scalar example to a real PDE benchmark, replace the
# scalar u with a vector u of length n on a 1D mesh, add an n x n
# tridiagonal Laplacian for diffusion, and apply lift_system to the
# resulting Vector{Num}. The lift step adds n auxiliary variables (one
# w_i = u_i^2 per grid point), and POD reduces the resulting 2n-dim
# lifted state down to r << 2n. That is the experiment in
# benchmarks/allen_cahn_benchmark.jl, and it is where the QuadraticTensor
# representation pays off: the dense H matrix would be n x n^2 entries
# but is mostly zero.

println("\nDone.")
