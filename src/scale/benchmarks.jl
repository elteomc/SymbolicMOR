# src/scale/benchmarks.jl
#
# Phase 3 - Speedup Benchmarking
#
# Run this script as:
#   julia --project=. -p 4 scripts/scaling_benchmark.jl
#
# The -p N flag launches Julia with N worker processes.
# Compare results across different values of N to get the speedup curve.

using Distributed
using BenchmarkTools

"""
    benchmark_serial_vs_parallel(f!, u0_ensemble, tspan; dt, n_samples)

Benchmark serial vs parallel snapshot generation using however many
workers are currently available (set via -p N at Julia startup).

Returns a NamedTuple with timing and speedup.
"""
function benchmark_serial_vs_parallel(
    f!,
    u0_ensemble::Vector{<:AbstractVector},
    tspan::Tuple{<:Real,<:Real};
    dt::Float64 = 0.01,
    n_samples::Int = 3
)
    n_workers = nworkers()

    # serial baseline - always single process
    t_serial = minimum(
        @elapsed generate_snapshots(f!, u0_ensemble, tspan; dt)
        for _ in 1:n_samples
    )

    # parallel - uses all available workers
    t_parallel = if n_workers == 1
        @warn "No extra workers available. Launch Julia with -p N for parallel benchmark."
        t_serial
    else
        minimum(
            @elapsed generate_snapshots_parallel(f!, u0_ensemble, tspan; dt)
            for _ in 1:n_samples
        )
    end

    speedup = t_serial / t_parallel

    @info "Workers: $n_workers"
    @info "Serial:   $(round(t_serial,   digits=3)) s"
    @info "Parallel: $(round(t_parallel, digits=3)) s"
    @info "Speedup:  $(round(speedup,    digits=2)) x"

    return (
        n_workers  = n_workers,
        t_serial   = t_serial,
        t_parallel = t_parallel,
        speedup    = speedup
    )
end
