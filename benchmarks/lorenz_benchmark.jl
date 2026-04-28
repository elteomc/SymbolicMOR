# Lorenz: (already quadratic) exercises snapshots, POD, operators, optional scaling.
#
# Run from repo root:
#   julia --project=. benchmarks/lorenz_benchmark.jl
#
# Parallel scaling (optional):
#   julia --project=. -p 4 benchmarks/lorenz_benchmark.jl

using OrdinaryDiffEq
using LinearAlgebra
using Random
using Symbolics
using SymbolicMOR

include(joinpath(@__DIR__, "common.jl"))

println("="^60)
println(" Benchmark: Lorenz (baseline full pipeline)")
println("="^60)

const sigma = 10.0
const rho = 28.0
const beta = 8 / 3

@variables x y z
rhs_lorenz = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
ls = lift_system([x, y, z], rhs_lorenz)

println("\n[1] Lift: aux vars introduced = ", length(ls.aux_eqs))

A, H, c = extract_operators(ls)

function lorenz!(du, u, p, t)
    du[1] = sigma * (u[2] - u[1])
    du[2] = u[1] * (rho - u[3]) - u[2]
    du[3] = u[1] * u[2] - beta * u[3]
    return nothing
end

t_train = (0.0, 8.0)
dt = 0.02
Random.seed!(42)
u0s = [randn(3) .* 0.4 .+ [1.0, 0.0, 25.0] for _ in 1:24]

println("\n[2] Snapshot matrix (serial)...")
X = generate_snapshots(lorenz!, u0s, t_train; dt)
println("    size(X) = ", size(X))

r = 3
Phi, sigma_sv, energy = compute_pod_basis(X, r)
println("\n[3] POD: retained r = $r, energy fraction approx ", round(energy * 100; digits=4), "%")

A_hat, H_hat, c_hat = galerkin_project(A, H, c, Phi)

u0_test = [1.0, 0.0, 25.0]
t_test = (0.0, 2.0)
a0 = Phi' * u0_test

prob_rom = ODEProblem(rom_rhs!, a0, t_test, (A_hat, H_hat, c_hat))
sol_rom = solve(prob_rom, Tsit5(); saveat=dt, abstol=1e-8, reltol=1e-8)

prob_full = ODEProblem(lorenz!, u0_test, t_test)
sol_full = solve(prob_full, Tsit5(); saveat=dt, abstol=1e-10, reltol=1e-10)

s_proj = [Phi * sol_rom.u[i] for i in eachindex(sol_rom.t)]
errs = [norm(s_proj[i] - sol_full.u[i]) for i in eachindex(sol_full.t)]
println("\n[4] ROM vs FOM (orthogonal projection of ROM state): max error over window = ", maximum(errs))

println("\n[5] Serial vs parallel snapshot timing (small ensemble)...")
Random.seed!(7)
u0_small = [randn(3) .* 0.5 .+ [1.0, 0.0, 25.0] for _ in 1:32]
results = benchmark_serial_vs_parallel(lorenz!, u0_small, (0.0, 3.0); dt=0.02, n_samples=2)
println("    Workers = $(results.n_workers), speedup approx $(round(results.speedup; digits=2)) x")

println("\nDone.")
