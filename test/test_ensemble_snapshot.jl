using Test
using SymbolicMOR

@testset "SciML ensemble snapshots" begin
    function linear_decay!(du, u, p, t)
        du[1] = -u[1]
        du[2] = -2u[2]
        return nothing
    end

    u0s = [[1.0, 0.5], [0.25, -0.75], [-0.5, 1.0]]
    tspan = (0.0, 0.2)

    X_serial = generate_snapshots(linear_decay!, u0s, tspan; dt=0.1, abstol=1e-10, reltol=1e-10)
    X_ensemble = generate_snapshots_ensemble(linear_decay!, u0s, tspan; dt=0.1, abstol=1e-10, reltol=1e-10)

    @test size(X_ensemble) == size(X_serial)
    @test isapprox(X_ensemble, X_serial; atol=1e-10, rtol=1e-10)

    @test_throws ArgumentError generate_snapshots_ensemble(linear_decay!, Vector{Vector{Float64}}(), tspan)
end
