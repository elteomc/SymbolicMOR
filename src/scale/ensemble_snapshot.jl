# src/scale/ensemble_snapshot.jl
#
# SciML EnsembleProblem-based snapshot generation. This keeps the public
# snapshot API close to SciML's ensemble interface while returning the same
# snapshot matrix layout as generate_snapshots.

using OrdinaryDiffEq
using LinearAlgebra

"""
    generate_snapshots_ensemble(f!, u0_ensemble, tspan; dt=0.01, solver=Tsit5(),
                                ensemble_alg=EnsembleSerial(), kwargs...)

Generate a snapshot matrix using SciML's `EnsembleProblem` interface.

Each trajectory starts from one initial condition in `u0_ensemble`. The return
value has the same layout as `generate_snapshots`: state dimension by total
number of saved snapshots.

Use `ensemble_alg=EnsembleThreads()` for a threaded run, or keep the default
`EnsembleSerial()` for deterministic serial behavior.
"""
function generate_snapshots_ensemble(
    f!,
    u0_ensemble::Vector{<:AbstractVector},
    tspan::Tuple{<:Real,<:Real};
    dt::Float64 = 0.01,
    solver = Tsit5(),
    ensemble_alg = EnsembleSerial(),
    kwargs...
)
    isempty(u0_ensemble) && throw(ArgumentError("u0_ensemble must not be empty"))

    base_prob = ODEProblem(f!, u0_ensemble[1], tspan)
    prob_func = (prob, i, repeat) -> ODEProblem(f!, u0_ensemble[i], tspan)
    ensemble_prob = EnsembleProblem(base_prob; prob_func)

    sol = solve(
        ensemble_prob,
        solver,
        ensemble_alg;
        trajectories = length(u0_ensemble),
        saveat = dt,
        kwargs...
    )

    return hcat((hcat(traj.u...) for traj in sol.u)...)
end
