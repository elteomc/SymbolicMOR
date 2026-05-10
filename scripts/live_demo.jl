# scripts/live_demo.jl
#
# Talk-day live demo. Runs the full SymbolicMOR pipeline on a small example
# the audience can follow in real time. Each section pauses for the speaker
# to explain what just happened.
#
# Usage:
#   julia --project=. -i scripts/live_demo.jl
#
# The `-i` flag drops into the REPL after the script finishes so the speaker
# can run extra ad-hoc commands. Set the env var SYMOR_DEMO_AUTO=1 to skip
# the pauses (useful for dry-runs and CI).

using Symbolics
using SymbolicMOR
using OrdinaryDiffEq
using LinearAlgebra
using Random

const _AUTO = get(ENV, "SYMOR_DEMO_AUTO", "0") == "1"

function pause(label::AbstractString = "next")
    _AUTO && return
    print("\n[press Enter for $label]")
    readline()
end

function banner(title)
    println()
    println("=" ^ 70)
    println("  ", title)
    println("=" ^ 70)
end

# ----------------------------------------------------------------------
banner("1. Pick a non-polynomial ODE: dx/dt = -tanh(x)")
# ----------------------------------------------------------------------

# A first-year grad student knows tanh: it has nonzero derivatives of all
# orders and is famously hard for closed-form intrusive MOR. SymbolicMOR
# rewrites it into a polynomial system using one identity:
#
#       d/dt tanh(g) = (1 - tanh(g)^2) * g_dot
#
# So with p = tanh(x), the lifted dynamics are quadratic:
#
#       dx/dt = -p
#       dp/dt = -(1 - p^2) * p   = -p + p^3
#
# (and p^3 then gets quadratized in the next phase).

@variables x
rhs = [-tanh(x)]
println("Original RHS: ", rhs[1])

pause("the lift")

# ----------------------------------------------------------------------
banner("2. Symbolic lift")
# ----------------------------------------------------------------------

ls = lift_system([x], rhs)
println("Aux equations introduced:")
for eq in ls.aux_eqs
    println("  ", eq.lhs, " = ", eq.rhs)
end
println()
println("Lifted state z = ", ls.lifted_vars)
println("Polynomial RHS G(z):")
for (v, f) in zip(ls.lifted_vars, ls.F)
    println("  d", v, "/dt = ", f)
end

pause("snapshot generation")

# ----------------------------------------------------------------------
banner("3. Snapshots")
# ----------------------------------------------------------------------

# Compile the lifted vector field once. This is what populates the snapshot
# matrix from many initial conditions.
G! = build_lifted_rhs(ls)

# Build initial conditions in the lifted state by evaluating each aux
# definition in order (so chained auxes like p^2 see the right numeric p).
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

Random.seed!(0xCAFE)
u0_orig_set = [[v] for v in 0.5 .+ 0.5 .* randn(20)]
u0_lift_set = [lifted_ic(ls, u0o) for u0o in u0_orig_set]

X = generate_snapshots(G!, u0_lift_set, (0.0, 5.0); dt=0.05)
println("Snapshot matrix size: ", size(X, 1), " x ", size(X, 2),
        " (lifted_dim x total_steps)")

pause("POD basis")

# ----------------------------------------------------------------------
banner("4. POD basis")
# ----------------------------------------------------------------------

# Pick r = 2 to get a visibly small ROM. The energy retained tells us how
# well two modes capture the snapshot data.
V, sigma, energy = compute_pod_basis(X, 2)
println("Singular values (first five): ", round.(sigma[1:min(5, length(sigma))]; sigdigits=4))
println("Energy retained by r = 2 modes: ", round(energy * 100; digits=4), "%")

pause("two ROM forms side by side")

# ----------------------------------------------------------------------
banner("5. Build both ROM forms and compare them")
# ----------------------------------------------------------------------

# Simple form: dot a = V' G(V a). One line.
rom_simple! = build_simple_rom_rhs(ls, V)

# Explicit form: dot a = A_hat a + H_hat (a kron a) + c_hat. Pre-computed.
A, Q, c = extract_quadratic_tensor(ls)
A_hat, Q_hat, c_hat = galerkin_project(A, Q, c, V)

# Reduced initial condition for one trajectory.
a0 = V' * u0_lift_set[1]

prob_simple   = ODEProblem(rom_simple!, a0, (0.0, 5.0))
prob_explicit = ODEProblem(rom_rhs!, a0, (0.0, 5.0), (A_hat, Q_hat, c_hat))

sol_simple   = solve(prob_simple,   Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
sol_explicit = solve(prob_explicit, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

err_form = maximum(norm(sol_simple.u[i] - sol_explicit.u[i]) for i in eachindex(sol_simple.t))
println("Max difference between the two ROM forms over the trajectory: ", err_form)
println("(Should be at solver tolerance.)")

pause("the comparison")

# ----------------------------------------------------------------------
banner("6. ROM vs full lifted simulation")
# ----------------------------------------------------------------------

# Reconstruct ROM trajectory in the lifted space and compare to the full
# lifted simulation from the same IC. The first component is x.
prob_full = ODEProblem(G!, u0_lift_set[1], (0.0, 5.0))
sol_full  = solve(prob_full, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)

reconstructed = [V * a for a in sol_explicit.u]
err_x = maximum(abs(reconstructed[i][1] - sol_full.u[i][1]) for i in eachindex(sol_full.t))
println("Reduced state dim r = ", size(V, 2),
        ", lifted dim N = ", length(ls.lifted_vars))
println("Max |x_ROM - x_full| over the trajectory: ", err_x)
println()

if size(V, 2) >= length(ls.lifted_vars)
    println("With r = N the ROM is exact up to solver tolerance. To see")
    println("a real reduction, drop r below the lifted dimension.")
end

pause("the closing slide")

# ----------------------------------------------------------------------
banner("Done. The REPL is yours.")
# ----------------------------------------------------------------------

println("Variables left in scope (try them):")
println("  ls            : the LiftedSystem")
println("  V             : POD basis (N x r)")
println("  rom_simple!   : simple Galerkin RHS closure")
println("  A_hat, Q_hat, c_hat : explicit ROM operators")
println("  sol_full, sol_explicit : the two trajectories")
println()
println("Try:")
println("  ls2 = lift_system([x], [exp(-x)])    # different non-polynomial atom")
println("  ls3 = lift_system([x], [sin(x)])     # the sin/cos closure")
