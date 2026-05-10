# Portable benchmark campaign runner for local machines and HPC jobs.
#
# Examples:
#   julia --project=. -p 1 benchmarks/cluster_campaign.jl --case lorenz --trajectories 64 --out benchmark_results/local_p1.csv
#   julia --project=. -p 4 benchmarks/cluster_campaign.jl --case lorenz --trajectories 512 --tend 10 --dt 0.02 --repeats 3 --out benchmark_results/local_p4.csv

using Distributed
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Random
using Statistics
using SymbolicMOR

@everywhere using OrdinaryDiffEq
@everywhere using SymbolicMOR

function parse_args(args)
    config = Dict(
        "case" => "lorenz",
        "trajectories" => "64",
        "tend" => "5.0",
        "dt" => "0.02",
        "repeats" => "3",
        "out" => "benchmark_results/campaign.csv",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            i == length(args) && error("missing value for argument $arg")
            config[key] = args[i + 1]
            i += 2
        else
            error("unexpected argument: $arg")
        end
    end

    return (
        case = config["case"],
        trajectories = parse(Int, config["trajectories"]),
        tend = parse(Float64, config["tend"]),
        dt = parse(Float64, config["dt"]),
        repeats = parse(Int, config["repeats"]),
        out = config["out"],
    )
end

function print_help()
    println("""
    Usage:
      julia --project=. -p N benchmarks/cluster_campaign.jl [options]

    Options:
      --case CASE             lorenz, cubic, allen_cahn, random_quadratic
      --trajectories N        number of independent trajectories
      --tend T                final time
      --dt DT                 save interval
      --repeats N             timing repeats
      --out PATH              CSV output path
    """)
end

@everywhere function campaign_lorenz!(du, u, p, t)
    sigma, rho, beta = 10.0, 28.0, 8 / 3
    du[1] = sigma * (u[2] - u[1])
    du[2] = u[1] * (rho - u[3]) - u[2]
    du[3] = u[1] * u[2] - beta * u[3]
    return nothing
end

@everywhere function campaign_cubic!(du, u, p, t)
    du[1] = -u[1]^3
    return nothing
end

@everywhere function campaign_allen_cahn_smoke!(du, u, p, t)
    du[1] = -u[1]^3
    return nothing
end

@everywhere function campaign_random_quadratic!(du, u, p, t)
    A, Q, c = p
    du .= A * u .+ c
    evaluate_quadratic!(du, Q, u; reset = false)
    return nothing
end

function build_case(case_name::String, trajectories::Int)
    Random.seed!(123)

    if case_name == "lorenz"
        u0s = [randn(3) .* 0.5 .+ [1.0, 0.0, 25.0] for _ in 1:trajectories]
        return campaign_lorenz!, u0s, nothing
    elseif case_name == "cubic"
        u0s = [[0.5 + rand()] for _ in 1:trajectories]
        return campaign_cubic!, u0s, nothing
    elseif case_name == "allen_cahn"
        u0s = [[0.5 + rand()] for _ in 1:trajectories]
        return campaign_allen_cahn_smoke!, u0s, nothing
    elseif case_name == "random_quadratic"
        n = 24
        n_terms = 4n
        Q = QuadraticTensor(n, n, rand(1:n, n_terms), rand(1:n, n_terms), rand(1:n, n_terms), randn(n_terms))
        A = -0.05I(n)
        c = zeros(n)
        params = (A, Q, c)
        f! = (du, u, p, t) -> campaign_random_quadratic!(du, u, params, t)
        u0s = [0.05 .* randn(n) for _ in 1:trajectories]
        return f!, u0s, params
    else
        error("unknown case '$case_name'; expected lorenz, cubic, allen_cahn, or random_quadratic")
    end
end

function min_elapsed(f, repeats::Int)
    return minimum(@elapsed f() for _ in 1:repeats)
end

function git_sha()
    try
        return strip(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function cpu_label()
    try
        return Sys.cpu_info()[1].model
    catch
        return "unknown"
    end
end

function write_csv(path::String, row::NamedTuple)
    mkpath(dirname(path))
    headers = collect(keys(row))
    values = [string(getfield(row, key)) for key in headers]
    new_file = !isfile(path)

    open(path, "a") do io
        new_file && println(io, join(headers, ","))
        println(io, join(values, ","))
    end
end

function run_campaign(config)
    f!, u0s, _ = build_case(config.case, config.trajectories)
    tspan = (0.0, config.tend)
    workers = nworkers()

    serial_time = min_elapsed(config.repeats) do
        generate_snapshots(f!, u0s, tspan; dt = config.dt)
    end

    parallel_time = if workers == 1
        serial_time
    else
        min_elapsed(config.repeats) do
            generate_snapshots_parallel(f!, u0s, tspan; dt = config.dt)
        end
    end

    speedup = serial_time / parallel_time
    efficiency = workers == 0 ? speedup : speedup / workers

    row = (
        case = config.case,
        workers = workers,
        trajectories = config.trajectories,
        tend = config.tend,
        dt = config.dt,
        repeats = config.repeats,
        serial_time = serial_time,
        parallel_time = parallel_time,
        speedup = speedup,
        efficiency = efficiency,
        julia_version = string(VERSION),
        os = string(Sys.KERNEL),
        cpu = replace(cpu_label(), "," => ";"),
        git_sha = git_sha(),
    )

    write_csv(config.out, row)

    @printf("case=%s workers=%d trajectories=%d serial=%.4f parallel=%.4f speedup=%.3f efficiency=%.3f\n",
        row.case, row.workers, row.trajectories, row.serial_time, row.parallel_time, row.speedup, row.efficiency)
    println("wrote ", config.out)

    return row
end

config = parse_args(ARGS)
run_campaign(config)
