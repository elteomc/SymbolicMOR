# Summarize CSV files produced by benchmarks/cluster_campaign.jl.
#
# Example:
#   julia --project=. benchmarks/summarize_campaign.jl benchmark_results summary

using Plots
using Statistics

function parse_csv(path::String)
    lines = readlines(path)
    length(lines) < 2 && return NamedTuple[]

    headers = split(lines[1], ",")
    rows = NamedTuple[]

    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ",")
        data = Dict(headers[i] => values[i] for i in eachindex(headers))
        push!(rows, (
            case = data["case"],
            workers = parse(Int, data["workers"]),
            trajectories = parse(Int, data["trajectories"]),
            tend = parse(Float64, data["tend"]),
            dt = parse(Float64, data["dt"]),
            repeats = parse(Int, data["repeats"]),
            serial_time = parse(Float64, data["serial_time"]),
            parallel_time = parse(Float64, data["parallel_time"]),
            speedup = parse(Float64, data["speedup"]),
            efficiency = parse(Float64, data["efficiency"]),
            julia_version = data["julia_version"],
            os = data["os"],
            cpu = data["cpu"],
            git_sha = data["git_sha"],
        ))
    end

    return rows
end

function collect_rows(input_path::String)
    if isdir(input_path)
        files = filter(path -> endswith(path, ".csv"), readdir(input_path; join=true))
        return reduce(vcat, parse_csv.(files); init=NamedTuple[])
    else
        return parse_csv(input_path)
    end
end

function summarize(rows, output_prefix::String)
    isempty(rows) && error("no campaign rows found")
    mkpath(dirname(output_prefix))

    cases = sort(unique(row.case for row in rows))
    speedup_plot = plot(title = "SymbolicMOR benchmark speedup", xlabel = "Workers", ylabel = "Speedup")
    efficiency_plot = plot(title = "SymbolicMOR benchmark efficiency", xlabel = "Workers", ylabel = "Efficiency")

    for case_name in cases
        case_rows = sort(filter(row -> row.case == case_name, rows); by = row -> row.workers)
        workers = [row.workers for row in case_rows]
        speedups = [row.speedup for row in case_rows]
        efficiencies = [row.efficiency for row in case_rows]

        plot!(speedup_plot, workers, speedups; marker = :circle, label = case_name)
        plot!(efficiency_plot, workers, efficiencies; marker = :circle, label = case_name)
    end

    speedup_path = output_prefix * "_speedup.png"
    efficiency_path = output_prefix * "_efficiency.png"
    savefig(speedup_plot, speedup_path)
    savefig(efficiency_plot, efficiency_path)

    println("Rows: ", length(rows))
    println("Cases: ", join(cases, ", "))
    println("Wrote ", speedup_path)
    println("Wrote ", efficiency_path)
end

if length(ARGS) < 1 || length(ARGS) > 2
    println("Usage: julia --project=. benchmarks/summarize_campaign.jl INPUT_CSV_OR_DIR [OUTPUT_PREFIX]")
    exit(1)
end

input_path = ARGS[1]
output_prefix = length(ARGS) == 2 ? ARGS[2] : "benchmark_results/summary"
summarize(collect_rows(input_path), output_prefix)
