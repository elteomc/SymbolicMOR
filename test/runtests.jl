using Test
using SymbolicMOR

@testset "SymbolicMOR.jl" begin
    include("test_quadratize.jl")
    include("test_pod.jl")
    include("test_quadratic_operator.jl")
    include("test_lorenz.jl")
    include("test_ensemble_snapshot.jl")
    include("test_mtk_api.jl")
end
