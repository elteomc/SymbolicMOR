### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721001
begin
    pushfirst!(LOAD_PATH, normpath(@__DIR__, ".."))
    using SymbolicMOR
    using Symbolics
    using ModelingToolkit
    using OrdinaryDiffEq
    using LinearAlgebra
    using Random
    using Statistics
    using SparseArrays
    using PlutoUI
    using Plots
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721023
macro bind(def, element)
    return quote
        local widget = $(esc(element))
        global $(esc(def)) = try
            Base.get(widget)
        catch
            missing
        end
        widget
    end
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721002
md"""
# SymbolicMOR Current State

This notebook is a guided technical walkthrough of the project as it exists
now. Change the controls, rerun cells, and inspect
intermediate objects.

The pipeline is:

1. define polynomial dynamics,
2. symbolically lift them to quadratic form, including a narrow MTK path,
3. generate trajectory snapshots,
4. compute a POD basis,
5. extract dense, sparse, or tensor quadratic operators,
6. project to a reduced model,
7. compare behavior and discuss scaling tools.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721003
TableOfContents()

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721004
md"""
## 1. Lorenz: the quadratic baseline

Lorenz is already quadratic. That makes it a useful sanity check: the lift
should preserve the original state dimension and introduce no auxiliary
variables.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721005
begin
    sigma_val = 10.0
    rho_val = 28.0
    beta_val = 8 / 3

    @variables x y z
    lorenz_rhs = [
        sigma_val * (y - x),
        x * (rho_val - z) - y,
        x * y - beta_val * z,
    ]

    lorenz_lifted = lift_system([x, y, z], lorenz_rhs)
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721006
begin
    lorenz_summary = (
        original_states = 3,
        lifted_states = length(lorenz_lifted.lifted_vars),
        auxiliary_equations = length(lorenz_lifted.aux_eqs),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721024
md"""
### ModelingToolkit entry point

The vector API remains the core path, but explicit ModelingToolkit
`ODESystem`s can now be adapted into the same lifting pipeline.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721025
begin
    @independent_variables t_mtk
    @variables x_mtk(t_mtk) y_mtk(t_mtk) z_mtk(t_mtk)
    D_mtk = Differential(t_mtk)
    lorenz_mtk_rhs = [
        sigma_val * (y_mtk - x_mtk),
        x_mtk * (rho_val - z_mtk) - y_mtk,
        x_mtk * y_mtk - beta_val * z_mtk,
    ]
    @named lorenz_sys = ODESystem(D_mtk.([x_mtk, y_mtk, z_mtk]) .~ lorenz_mtk_rhs, t_mtk)

    mtk_states, mtk_rhs = extract_state_rhs(lorenz_sys)
    mtk_lifted = lift_system(lorenz_sys)
    mtk_summary = (
        states = mtk_states,
        rhs_count = length(mtk_rhs),
        lifted_states = length(mtk_lifted.lifted_vars),
        auxiliary_equations = length(mtk_lifted.aux_eqs),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721007
md"""
## 2. Cubic decay: why lifting matters

The scalar equation `du/dt = -u^3` is not quadratic in the original variable.
The lift introduces auxiliary state so the dynamics can be represented with
linear, quadratic, and constant operators.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721008
begin
    @variables u
    cubic_lifted = lift_system([u], [-u^3])
    cubic_summary = (
        original_states = 1,
        lifted_states = length(cubic_lifted.lifted_vars),
        auxiliary_equations = length(cubic_lifted.aux_eqs),
        lifted_variables = cubic_lifted.lifted_vars,
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721026
begin
    function degree_summary(ls)
        [maximum(Symbolics.degree(expr, var) for var in ls.lifted_vars; init = 0) for expr in ls.F]
    end

    aux_inspector = (
        cubic_aux_definitions = cubic_lifted.aux_eqs,
        cubic_lifted_rhs = cubic_lifted.F,
        cubic_rhs_degrees = degree_summary(cubic_lifted),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721009
md"""
## 3. Snapshots and POD

These controls keep the live demo small. Increasing trajectories, horizon, or
rank makes the demo more representative but also slower.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721010
md"""
Training trajectories: $(@bind n_train Slider(4:4:40, default=12, show_value=true))

Training horizon: $(@bind train_end Slider(1.0:0.5:8.0, default=3.0, show_value=true))

Snapshot spacing: $(@bind save_dt Slider(0.01:0.01:0.08, default=0.03, show_value=true))

POD rank: $(@bind pod_rank Slider(1:3, default=3, show_value=true))
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721011
function lorenz!(du, state, params, t)
    du[1] = sigma_val * (state[2] - state[1])
    du[2] = state[1] * (rho_val - state[3]) - state[2]
    du[3] = state[1] * state[2] - beta_val * state[3]
    return nothing
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721012
begin
    Random.seed!(11)
    u0s = [randn(3) .* 0.35 .+ [1.0, 0.0, 25.0] for _ in 1:n_train]
    X = generate_snapshots(lorenz!, u0s, (0.0, train_end); dt=save_dt)
    Phi, singular_values, energy = compute_pod_basis(X, pod_rank)
    snapshot_summary = (
        snapshot_size = size(X),
        retained_rank = size(Phi, 2),
        energy_percent = round(100 * energy; digits=4),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721013
plot(
    singular_values;
    yscale = :log10,
    xlabel = "Mode index",
    ylabel = "Singular value",
    title = "POD singular value decay",
    marker = :circle,
    legend = false,
)

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721014
md"""
## 4. Operator representations

The compatibility path returns dense `A, H, c` operators. The newer paths also
support sparse matrix storage and `QuadraticTensor`, which avoids forming
`kron(x, x)` during evaluation.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721015
begin
    A_dense, H_dense, c_dense = extract_operators(lorenz_lifted)
    A_sparse, H_sparse, c_sparse = extract_operators_sparse(lorenz_lifted)
    A_tensor, Q_tensor, c_tensor = extract_quadratic_tensor(lorenz_lifted)

    operator_summary = (
        A_size = size(A_dense),
        H_size = size(H_dense),
        dense_H_nonzeros = count(!iszero, H_dense),
        sparse_H_nonzeros = nnz(H_sparse),
        tensor_terms = length(Q_tensor),
        sparse_matches_dense = Matrix(H_sparse) == H_dense,
        tensor_matches_dense = dense_matrix(Q_tensor) == H_dense,
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721027
begin
    tensor_coordinate_preview = [
        (
            row = Q_tensor.rows[i],
            col1 = Q_tensor.cols1[i],
            col2 = Q_tensor.cols2[i],
            value = Q_tensor.vals[i],
        )
        for i in 1:min(length(Q_tensor), 8)
    ]
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721016
md"""
## 5. ROM projection and short-horizon comparison

Chaotic systems should not be judged by long-time pointwise agreement. Here the
goal is a short-window smoke check of the ROM pipeline.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721017
begin
    A_hat_dense, H_hat_dense, c_hat_dense = galerkin_project(A_dense, H_dense, c_dense, Phi)
    A_hat_tensor, Q_hat_tensor, c_hat_tensor = galerkin_project(A_tensor, Q_tensor, c_tensor, Phi)

    u0_test = [1.0, 0.0, 25.0]
    a0 = Phi' * u0_test
    compare_tspan = (0.0, min(2.0, train_end))

    full_prob = ODEProblem(lorenz!, u0_test, compare_tspan)
    full_sol = solve(full_prob, Tsit5(); saveat=save_dt, abstol=1e-10, reltol=1e-10)

    rom_prob = ODEProblem(rom_rhs!, a0, compare_tspan, (A_hat_tensor, Q_hat_tensor, c_hat_tensor))
    rom_sol = solve(rom_prob, Tsit5(); saveat=save_dt, abstol=1e-8, reltol=1e-8)

    reconstructed = [Phi * rom_sol.u[i] for i in eachindex(rom_sol.u)]
    errors = [norm(reconstructed[i] - full_sol.u[i]) for i in eachindex(reconstructed)]
    rom_summary = (
        max_error = maximum(errors),
        final_error = errors[end],
        dense_tensor_projection_agree = isapprox(
            dense_matrix(Q_hat_tensor),
            H_hat_dense;
            atol = 1e-10,
            rtol = 1e-10,
        ),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721018
plot(
    full_sol.t,
    errors;
    xlabel = "t",
    ylabel = "Reconstruction error",
    title = "Short-horizon ROM error",
    marker = :circle,
    legend = false,
)

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721019
md"""
## 6. Scaling tools now in the project

There are three practical scaling hooks:

* `generate_snapshots_parallel` distributes independent trajectories with
  Julia `Distributed`.
* `generate_snapshots_ensemble` uses SciML `EnsembleProblem`.
* `QuadraticTensor` evaluates quadratic terms without building `kron(x, x)`.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721020
begin
    tensor_rhs_time = @elapsed begin
        buffer = similar(a0)
        for _ in 1:500
            rom_rhs!(buffer, a0, (A_hat_tensor, Q_hat_tensor, c_hat_tensor), 0.0)
        end
    end

    dense_rhs_time = @elapsed begin
        buffer = similar(a0)
        for _ in 1:500
            rom_rhs!(buffer, a0, (A_hat_dense, H_hat_dense, c_hat_dense), 0.0)
        end
    end

    scaling_summary = (
        dense_rhs_500_calls_seconds = round(dense_rhs_time; sigdigits=4),
        tensor_rhs_500_calls_seconds = round(tensor_rhs_time; sigdigits=4),
        ensemble_helper_available = isdefined(SymbolicMOR, :generate_snapshots_ensemble),
        distributed_helper_available = isdefined(SymbolicMOR, :generate_snapshots_parallel),
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721028
begin
    dense_projection_time = @elapsed galerkin_project(A_dense, H_dense, c_dense, Phi)
    tensor_projection_time = @elapsed galerkin_project(A_tensor, Q_tensor, c_tensor, Phi)
    projection_timing_summary = (
        dense_projection_seconds = dense_projection_time,
        tensor_projection_seconds = tensor_projection_time,
    )
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721029
md"""
## 7. Campaign result import

If `benchmark_results` contains CSV files from `benchmarks/cluster_campaign.jl`,
this cell summarizes them. It is fine if no result files exist yet.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721030
begin
    function read_campaign_rows(dir)
        isdir(dir) || return NamedTuple[]
        files = filter(path -> endswith(path, ".csv"), readdir(dir; join = true))
        rows = NamedTuple[]

        for file in files
            lines = readlines(file)
            length(lines) < 2 && continue
            headers = split(lines[1], ",")
            for line in lines[2:end]
                isempty(strip(line)) && continue
                values = split(line, ",")
                data = Dict(headers[i] => values[i] for i in eachindex(headers))
                push!(rows, (
                    case = data["case"],
                    workers = parse(Int, data["workers"]),
                    speedup = parse(Float64, data["speedup"]),
                    efficiency = parse(Float64, data["efficiency"]),
                ))
            end
        end

        return rows
    end

    campaign_rows = read_campaign_rows(normpath(@__DIR__, "..", "benchmark_results"))
    campaign_summary = isempty(campaign_rows) ? "No campaign CSV files found yet." : campaign_rows
end

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721021
md"""
## 8. Current limitations

* The package is still a research prototype, not a registered Julia package.
* Polynomial dynamics are the supported path; broader polynomialization remains
  experimental.
* ModelingToolkit support is currently limited to explicit polynomial
  `ODESystem`s with one `D(x) ~ rhs` equation per state.
* Parallel snapshot speedups need large ensembles or cluster/Linux runs to be
  convincing.
* This notebook is intentionally small so it can run live with a colleague.
"""

# ╔═╡ 1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721022
md"""
## Discussion checklist

* Show `lorenz_summary` and explain why Lorenz is a clean baseline.
* Show `mtk_summary` to demonstrate the ModelingToolkit adapter.
* Show `cubic_summary` and explain what lifting buys you.
* Show `aux_inspector` to inspect auxiliary variables and lifted RHS degrees.
* Move the snapshot/POD controls and watch `snapshot_summary` change.
* Compare `operator_summary` across dense, sparse, and tensor forms.
* Inspect `tensor_coordinate_preview` for the sparse coordinate layout.
* Use `rom_summary` as a smoke check, not as a claim of long-time chaos
  prediction.
* End with `scaling_summary`, `projection_timing_summary`, campaign results,
  and the limitation list.
"""

# ╔═╡ Cell order:
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721001
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721023
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721002
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721003
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721004
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721005
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721006
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721024
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721025
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721007
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721008
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721026
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721009
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721010
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721011
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721012
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721013
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721014
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721015
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721027
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721016
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721017
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721018
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721019
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721020
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721028
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721029
# ╠═1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721030
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721021
# ╟─1f4b7c3e-8f6a-11ee-1a1a-3b1d4d721022
