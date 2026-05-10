# scripts/walkthrough_vanderpol.jl
#
# Narrated walkthrough: Van der Pol oscillator.
# Run with:
#   julia --project=. scripts/walkthrough_vanderpol.jl
#
# Van der Pol is the textbook nonlinear oscillator. It has a stable limit
# cycle that is not a circle, and the nonlinearity is a cubic mixed term
# x^2 * y. This makes it a classic test case for nonlinear MOR: the
# trajectory cannot be approximated well by a single linear basis vector,
# but a small POD basis (r approx 4) captures the limit cycle.

using Symbolics
using SymbolicMOR
using OrdinaryDiffEq
using LinearAlgebra
using Random

println("="^70)
println(" Van der Pol walkthrough:  d^2x/dt^2 - mu (1 - x^2) dx/dt + x = 0 ")
println("="^70)

# ---------------------------------------------------------------------
# 1. State-space form
# ---------------------------------------------------------------------
#
# Set y = dx/dt to put the system in first order:
#
#       dx/dt = y
#       dy/dt = mu (1 - x^2) y - x
#             = mu y - mu x^2 y - x
#
# The mu * x^2 * y term is degree 3 in (x, y). The lift will need at
# least one auxiliary to handle it.

const mu = 1.0

@variables x y
rhs = [y, mu * y - mu * x^2 * y - x]
ls = lift_system([x, y], rhs)

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
# Quadratization sees x^2 * y, factors it as x * x * y, and introduces
# w1 = x * x. Then x * x * y becomes w1 * y, which is already quadratic.
# So we expect one new aux variable, w1 = x^2, plus a chain-rule equation
# dw1/dt = 2 x dx/dt = 2 x y. That is again quadratic.

# ---------------------------------------------------------------------
# 3. Snapshots
# ---------------------------------------------------------------------
#
# Use a spread of initial conditions inside the limit cycle so the
# snapshots cover both the transient phase and the cycle itself. Without
# the transient the POD basis would describe only the cycle and a fresh
# initial condition would project poorly.

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

Random.seed!(7)
init_states = [randn(2) .* 0.4 .+ [2.0, 0.0] for _ in 1:24]
u0_lift_set = [lifted_ic(ls, u0o) for u0o in init_states]
X = generate_snapshots(G!, u0_lift_set, (0.0, 12.0); dt=0.05)
println("\nSnapshot matrix size: ", size(X))

# ---------------------------------------------------------------------
# 4. POD basis
# ---------------------------------------------------------------------
#
# Three modes typically capture more than 99 percent of the energy
# because the lifted state is just (x, y, w1) and x and w1 = x^2 are
# almost colinear along the cycle. Push r down to see the ROM start to
# break.

r = 2
V, sigma, energy = compute_pod_basis(X, r)
println("\nSingular values (first 5): ", round.(sigma[1:min(5, length(sigma))]; sigdigits=4))
println("Energy retained by r = $r modes: ", round(energy * 100; digits=4), "%")

# ---------------------------------------------------------------------
# 5. Build both ROM forms
# ---------------------------------------------------------------------

A, Q, c = extract_quadratic_tensor(ls)
A_hat, Q_hat, c_hat = galerkin_project(A, Q, c, V)
rom_simple! = build_simple_rom_rhs(ls, V)

# ---------------------------------------------------------------------
# 6. Test against the full simulation
# ---------------------------------------------------------------------

u0_full = lifted_ic(ls, [2.0, 0.0])
a0 = V' * u0_full
tspan = (0.0, 12.0)

sol_full     = solve(ODEProblem(G!, u0_full, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_exp  = solve(ODEProblem(rom_rhs!, a0, tspan, (A_hat, Q_hat, c_hat)), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_rom_simp = solve(ODEProblem(rom_simple!, a0, tspan), Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

err_form = maximum(norm(sol_rom_exp.u[i] - sol_rom_simp.u[i]) for i in eachindex(sol_rom_exp.t))
err_x    = maximum(abs((V * sol_rom_exp.u[i])[1] - sol_full.u[i][1]) for i in eachindex(sol_full.t))
err_y    = maximum(abs((V * sol_rom_exp.u[i])[2] - sol_full.u[i][2]) for i in eachindex(sol_full.t))

println("\nSimple vs explicit ROM (should be solver tol): ", err_form)
println("Reduced (r=$r) vs full lifted:")
println("  max |x_ROM - x_full|: ", err_x)
println("  max |y_ROM - y_full|: ", err_y)

# ---------------------------------------------------------------------
# 7. Closing observation
# ---------------------------------------------------------------------
#
# Van der Pol's limit cycle is a closed curve in the (x, y) plane, so a
# strict rank-2 reduction cannot be exact globally even though the true
# state has only two degrees of freedom. The error reflects how well two
# orthonormal directions in the lifted (x, y, w1) space can approximate
# the cycle. For tighter accuracy, raise r to 3.

println("\nDone.")
