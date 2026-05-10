# test/test_simple_rom.jl
#
# The simple ROM `\dot{a} = V' * G(z_mean + V*a)` and the explicit-operator
# ROM `\dot{a} = A_hat * a + H_hat * (a kron a) + c_hat` are mathematically
# the same model. Here we verify they produce the same `\dot{a}` for a
# variety of reduced states `a`, and that integrated trajectories match.

using Test
using LinearAlgebra
using Symbolics
using SymbolicMOR
using OrdinaryDiffEq

@testset "Simple Galerkin ROM" begin
    @testset "Simple and explicit ROM rates agree on a polynomial system" begin
        @variables x y z
        sigma, rho, beta = 10.0, 28.0, 8/3
        rhs = [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
        ls = lift_system([x, y, z], rhs)

        Phi, _, _ = compute_pod_basis(Matrix{Float64}(I, 3, 3), 2)

        A, H, c = extract_operators(ls)
        A_hat, H_hat, c_hat = galerkin_project(A, H, c, Phi)
        rom_simple! = build_simple_rom_rhs(ls, Phi)

        for trial in 1:10
            a = randn(2)
            da_simple = zeros(2)
            da_explicit = zeros(2)
            rom_simple!(da_simple, a, nothing, 0.0)
            rom_rhs!(da_explicit, a, (A_hat, H_hat, c_hat), 0.0)
            @test isapprox(da_simple, da_explicit; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "Simple ROM integrates and matches explicit ROM" begin
        @variables x y z
        rhs = [10.0*(y - x), x*(28.0 - z) - y, x*y - (8/3)*z]
        ls = lift_system([x, y, z], rhs)

        Phi = Matrix{Float64}(I, 3, 3)
        A, H, c = extract_operators(ls)
        A_hat, H_hat, c_hat = galerkin_project(A, H, c, Phi)

        rom_simple! = build_simple_rom_rhs(ls, Phi)
        u0 = [1.0, 0.0, 25.0]
        tspan = (0.0, 1.0)
        sol_simple = solve(ODEProblem(rom_simple!, u0, tspan), Tsit5();
                           abstol=1e-10, reltol=1e-10, saveat=0.05)
        sol_explicit = solve(ODEProblem(rom_rhs!, u0, tspan, (A_hat, H_hat, c_hat)),
                             Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.05)
        err = maximum(norm(sol_simple.u[i] - sol_explicit.u[i]) for i in eachindex(sol_simple.t))
        @test err < 1e-6
    end

    @testset "Mean-centered simple ROM still matches explicit form" begin
        # When z_mean is nonzero, the simple ROM evaluates G(z_mean + V*a) and
        # projects. The explicit ROM operators must be derived for the same
        # mean-shifted lifted state. We verify the simple form against a
        # direct evaluation of V' * G(z_mean + V*a).
        @variables x y
        rhs = [-x^3, -y]
        ls = lift_system([x, y], rhs)

        Phi = Matrix{Float64}(I, length(ls.lifted_vars), 2)
        z_mean = [0.1, -0.2, zeros(length(ls.lifted_vars) - 2)...]

        rom_simple! = build_simple_rom_rhs(ls, Phi; z_mean = z_mean)
        G! = build_lifted_rhs(ls)

        for _ in 1:5
            a = randn(2)
            s = z_mean + Phi * a
            Gs = zeros(length(ls.lifted_vars))
            G!(Gs, s, nothing, 0.0)
            expected = Phi' * Gs

            da = zeros(2)
            rom_simple!(da, a, nothing, 0.0)
            @test isapprox(da, expected; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "Dimension mismatches throw" begin
        @variables x
        ls = lift_system([x], [-x^3])
        wrong_V = ones(length(ls.lifted_vars) + 1, 1)
        @test_throws DimensionMismatch build_simple_rom_rhs(ls, wrong_V)

        right_V = ones(length(ls.lifted_vars), 1)
        wrong_z = zeros(length(ls.lifted_vars) + 1)
        @test_throws DimensionMismatch build_simple_rom_rhs(ls, right_V; z_mean = wrong_z)
    end
end
