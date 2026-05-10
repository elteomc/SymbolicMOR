# scripts/walkthrough_lorenz.jl
#
# Narrated walkthrough: Lorenz '63 system.
# Run with:
#   julia --project=. scripts/walkthrough_lorenz.jl
#
# Lorenz is the famous chaotic toy model. It is also already quadratic in
# its state, which makes it a good sanity case for the lift pipeline: the
# lift step should add exactly zero auxiliary variables, and the ROM
# operators come straight from the original RHS. The interesting work is
# elsewhere, in the POD step and in the chaotic dynamics that POD sees.

using Symbolics
using SymbolicMOR
using OrdinaryDiffEq
using LinearAlgebra
using Random

println("="^70)
println(" Lorenz walkthrough:  three-dimensional chaotic flow ")
println("="^70)

# ---------------------------------------------------------------------
# 1. The system
# ---------------------------------------------------------------------
#
#       dx/dt = sigma (y - x)
#       dy/dt = x (rho - z) - y
#       dz/dt = x y - beta z
#
# The right-hand side is already a polynomial of degree two in (x, y, z).
# The three nonlinear products are -x*z, x*y, and the linear terms.

const sigma_p, rho_p, beta_p = 10.0, 28.0, 8/3

@variables x y z
rhs = [sigma_p * (y - x), x * (rho_p - z) - y, x * y - beta_p * z]
ls = lift_system([x, y, z], rhs)

println("\nOriginal state dim: ", length(ls.original_vars))
println("Lifted state dim:   ", length(ls.lifted_vars))
println("Aux equations introduced: ", length(ls.aux_eqs),
        "  (zero, as expected: Lorenz is already quadratic)")

# ---------------------------------------------------------------------
# 2. Why this matters
# ---------------------------------------------------------------------
#
# Many users encounter MOR through a system that is already polynomial.
# In that case the symbolic lift step is a no-op and the only work is
# extracting (A, H, c) from the symbolic RHS, computing a POD basis,
# and projecting. SymbolicMOR's design deliberately makes the lift step
# pay nothing extra in this case: lift_system returns immediately when
# the RHS is already polynomial.

# ---------------------------------------------------------------------
# 3. Snapshots
# ---------------------------------------------------------------------
#
# Lorenz is chaotic, so trajectories diverge exponentially from nearby
# initial conditions. We sample a small cloud of ICs near the standard
# attractor entry point and integrate for several Lyapunov times. The
# snapshot matrix grows linearly with the number of saved time points.

function lorenz!(du, u, p, t)
    du[1] = sigma_p * (u[2] - u[1])
    du[2] = u[1] * (rho_p - u[3]) - u[2]
    du[3] = u[1] * u[2] - beta_p * u[3]
    return nothing
end

Random.seed!(42)
ic_set = [randn(3) .* 0.4 .+ [1.0, 0.0, 25.0] for _ in 1:24]
X = generate_snapshots(lorenz!, ic_set, (0.0, 8.0); dt=0.02)
println("\nSnapshot matrix size: ", size(X))

# ---------------------------------------------------------------------
# 4. POD basis
# ---------------------------------------------------------------------
#
# For chaotic systems the singular values typically decay slowly: the
# attractor is fundamentally three-dimensional, so r = 3 is what we
# need. Picking r < 3 would leave large residuals.

V, sigma_sv, energy = compute_pod_basis(X, 3)
println("\nSingular values (first 5): ", round.(sigma_sv[1:min(5, length(sigma_sv))]; sigdigits=4))
println("Energy retained by r = 3 modes: ", round(energy * 100; digits=4), "%")

# ---------------------------------------------------------------------
# 5. Both ROM forms
# ---------------------------------------------------------------------

A, Q, c = extract_quadratic_tensor(ls)
A_hat, Q_hat, c_hat = galerkin_project(A, Q, c, V)
rom_simple! = build_simple_rom_rhs(ls, V)

# ---------------------------------------------------------------------
# 6. ROM vs full simulation
# ---------------------------------------------------------------------
#
# Lorenz being chaotic means the *trajectory* error eventually saturates
# at attractor diameter even for a perfect r = 3 ROM, because tiny
# integration differences amplify exponentially. We integrate a short
# horizon where exact tracking is meaningful.

u0_full = [1.0, 0.0, 25.0]
a0 = V' * u0_full
tspan = (0.0, 2.0)

sol_full     = solve(ODEProblem(lorenz!, u0_full, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.02)
sol_rom_exp  = solve(ODEProblem(rom_rhs!, a0, tspan, (A_hat, Q_hat, c_hat)), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.02)
sol_rom_simp = solve(ODEProblem(rom_simple!, a0, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.02)

err_form = maximum(norm(sol_rom_exp.u[i] - sol_rom_simp.u[i]) for i in eachindex(sol_rom_exp.t))
err_x    = maximum(abs((V * sol_rom_exp.u[i])[1] - sol_full.u[i][1]) for i in eachindex(sol_full.t))

println("\nSimple vs explicit ROM (should be solver tol): ", err_form)
println("Reduced (r=3) vs full FOM, x component over t in [0, 2]: ", err_x)

# ---------------------------------------------------------------------
# 7. Closing observation
# ---------------------------------------------------------------------
#
# With r equal to the original dimension the projection is bijective
# (within solver tolerance). This is not a useful reduction, of course,
# but it is the right correctness check. To see a *meaningful* reduction
# the same pipeline can be applied to a higher-dimensional system whose
# attractor still lives in a few effective dimensions. Lorenz is an
# illustration; Allen-Cahn (next walkthrough) is the more realistic case.

println("\nDone.")
