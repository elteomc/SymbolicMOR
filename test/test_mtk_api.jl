using Test
using ModelingToolkit
using SymbolicMOR
using Symbolics

@testset "ModelingToolkit ODESystem API" begin
    @testset "Lorenz ODESystem matches vector API" begin
        @independent_variables t
        @variables x(t) y(t) z(t)
        D = Differential(t)

        sigma, rho, beta = 10.0, 28.0, 8 / 3
        rhs = [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
        eqs = D.([x, y, z]) .~ rhs
        @named sys = ODESystem(eqs, t)

        states, sys_rhs = extract_state_rhs(sys)
        ls_mtk = lift_system(sys)
        ls_vec = lift_system([x, y, z], rhs)

        @test isequal(states, [x, y, z])
        @test isequal(sys_rhs, rhs)
        @test isequal(ls_mtk.original_vars, ls_vec.original_vars)
        @test isequal(ls_mtk.lifted_vars, ls_vec.lifted_vars)
        @test isequal(ls_mtk.F, ls_vec.F)
        @test length(ls_mtk.aux_eqs) == 0
    end

    @testset "Cubic ODESystem lifts through vector API" begin
        @independent_variables t
        @variables u(t)
        D = Differential(t)

        @named sys = ODESystem([D(u) ~ -u^3], t)
        ls_mtk = lift_system(sys)
        ls_vec = lift_system([u], [-u^3])

        @test isequal(ls_mtk.original_vars, [u])
        @test length(ls_mtk.lifted_vars) == length(ls_vec.lifted_vars)
        @test length(ls_mtk.aux_eqs) == length(ls_vec.aux_eqs)
        @test isequal(ls_mtk.F, ls_vec.F)
    end

    @testset "Unsupported MTK systems fail clearly" begin
        @independent_variables t
        @variables x(t) y(t)
        @parameters p
        D = Differential(t)

        @named dae_sys = ODESystem([D(x) ~ y, x ~ y], t)
        @test_throws ArgumentError extract_state_rhs(dae_sys)

        @named param_sys = ODESystem([D(x) ~ p * x], t)
        @test_throws ArgumentError extract_state_rhs(param_sys)
    end
end
