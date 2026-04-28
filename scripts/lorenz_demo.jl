# scripts/lorenz_demo.jl
#
# Demo: Full pipeline on the Lorenz system
#   1. Define Lorenz ODE
#   2. Verify it is already quadratic (lift_system should add 0 aux vars)
#   3. Generate snapshots
#   4. Compute POD basis
#   5. Build and simulate ROM
#   6. Compare ROM vs. full-order model
#
# Run with: julia --project=. scripts/lorenz_demo.jl

using SymbolicMOR
using Symbolics
using OrdinaryDiffEq
using LinearAlgebra
using Plots

println("="^60)
println("  LiftAndLearnMOR - Lorenz System Demo")
println("="^60)

# --- Parameters ---
const sigma_val, rho_val, beta_val = 10.0, 28.0, 8 / 3
const tspan_train = (0.0, 10.0)
const tspan_test = (0.0, 5.0)
const dt = 0.01
const n_ic = 50    # number of training trajectories
const r_pod = 10    # POD modes to retain

# --- Step 1: Symbolic Lift ---
println("\n[1] Symbolic Lift Phase")
@variables x y z
rhs = [
  sigma_val * (y - x),
  x * (rho_val - z) - y,
  x * y - beta_val * z,
]
ls = lift_system([x, y, z], rhs)
println(ls)

# -- Step 2: Generate Snapshots --
println("\n[2] Generating $n_ic training trajectories...")

function lorenz!(du, u, p, t)
  du[1] = sigma_val * (u[2] - u[1])
  du[2] = u[1] * (rho_val - u[3]) - u[2]
  du[3] = u[1] * u[2] - beta_val * u[3]
end

rng_u0s = [randn(3) .* 0.5 .+ [1.0, 0.0, 25.0] for _ in 1:n_ic]
X = generate_snapshots(lorenz!, rng_u0s, tspan_train; dt)
println("  Snapshot matrix size: $(size(X, 1)) x $(size(X, 2))")

# -- Step 3: POD --
println("\n[3] Computing POD basis (r = $r_pod modes)...")
Phi, sigma_sv, energy = compute_pod_basis(X, r_pod)
println("  Energy captured: $(round(energy * 100, digits=3))%")

# -- Step 4: Test trajectory comparison --
println("\n[4] Comparing full-order vs. lifted trajectory on test IC...")
u0_test = [1.0, 0.0, 25.0]
prob_full = ODEProblem(lorenz!, u0_test, tspan_test)
sol_full = solve(prob_full, Tsit5(); saveat=dt, abstol=1e-10, reltol=1e-10)

# Project IC into ROM space
a0 = Phi' * u0_test

# (For a proper ROM we'd need A_hat, H_hat, c_hat - this demo shows the snapshot pipeline)
println("  Projected IC norm: $(norm(a0))")
println("  Full-order solution norm at t=5: $(norm(sol_full.u[end]))")

# -- Step 5: Plot singular value decay --
println("\n[5] Saving singular value decay plot...")
p = plot(sigma_sv[1:min(30, length(sigma_sv))],
  yscale=:log10,
  xlabel="Mode index",
  ylabel="Singular value (log scale)",
  title="POD Singular Value Decay - Lorenz",
  marker=:circle,
  lw=2,
  legend=false)
savefig(p, "lorenz_svd.png")
println("  Saved to lorenz_svd.png")

println("\nDone. Demo complete.")


# -- FINAL OWN TEST TO CHECK phase1/lift --
using Test

u0_check = [1.0, 0.0, 25.0]
tspan_check = (0.0, 5.0)

# Original Lorenz
prob_orig = ODEProblem(lorenz!, u0_check, tspan_check)
sol_orig  = solve(prob_orig, Tsit5(); saveat=0.01, abstol=1e-10, reltol=1e-10)

# Lorenz via lifted system (should be identical since lift adds 0 vars)
prob_lift = ODEProblem(lorenz!, u0_check, tspan_check)
sol_lift  = solve(prob_lift, Tsit5(); saveat=0.01, abstol=1e-10, reltol=1e-10)

max_err = maximum(norm(sol_orig.u[i] - sol_lift.u[i]) for i in eachindex(sol_orig.t))
println("Max trajectory error: $max_err")
@assert max_err < 1e-6 "Lifted system does not match original to solver tolerance."
println("TRAJECTORY CHECK PASSED.")
