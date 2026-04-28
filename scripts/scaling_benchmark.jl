# scripts/scaling_benchmark.jl
#
# Run as: julia --project=. -p 1 scripts/scaling_benchmark.jl
#         julia --project=. -p 2 scripts/scaling_benchmark.jl
#         julia --project=. -p 4 scripts/scaling_benchmark.jl
#
# Collect the speedup numbers from each run to build your scaling plot.

using Distributed
using SymbolicMOR
# @everywhere using SymbolicMOR, OrdinaryDiffEq
@everywhere using OrdinaryDiffEq

@everywhere function lorenz!(du, u, p, t)
  sigma, rho, beta = 10.0, 28.0, 8 / 3
  du[1] = sigma * (u[2] - u[1])
  du[2] = u[1] * (rho - u[3]) - u[2]
  du[3] = u[1] * u[2] - beta * u[3]
end

#=
The work per trajectory is too small relative to the overhead of distributing it across workers.
Spinning up 4 workers and shipping data back and forth costs way more.

[ Info: Workers: 1
[ Info: Serial:   0.032 s
[ Info: Parallel: 0.032 s
[ Info: Speedup:  1.0 x
Done. Workers=1, Speedup=1.0x

[ Info: Workers: 2
[ Info: Serial:   0.133 s
[ Info: Parallel: 0.094 s
[ Info: Speedup:  1.41 x
Done. Workers=2, Speedup=1.41x

[ Info: Workers: 4
[ Info: Serial:   0.032 s
[ Info: Parallel: 0.037 s
[ Info: Speedup:  0.86 x
Done. Workers=4, Speedup=0.86x
=#
# u0s = [randn(3) .* 2 .+ [1.0, 0.0, 25.0] for _ in 1:100]
# tspan = (0.0, 5.0)

u0s = [randn(3) .* 2 .+ [1.0, 0.0, 25.0] for _ in 1:500]
tspan = (0.0, 20.0)

results = benchmark_serial_vs_parallel(lorenz!, u0s, tspan; dt=0.01, n_samples=3)
println("Done. Workers=$(results.n_workers), Speedup=$(round(results.speedup, digits=2))x")
