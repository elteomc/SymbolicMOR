using Test
using LinearAlgebra
using SparseArrays
using Symbolics
using SymbolicMOR

@testset "Quadratic operator representations" begin
    @testset "Dense, sparse, and tensor extraction agree" begin
        @variables x y
        ls = lift_system([x, y], [-x^3, -y])

        A_dense, H_dense, c_dense = extract_operators(ls)
        A_sparse, H_sparse, c_sparse = extract_operators_sparse(ls)
        A_tensor, Q, c_tensor = extract_quadratic_tensor(ls)

        @test A_tensor == A_dense
        @test c_tensor == c_dense
        @test Matrix(A_sparse) == A_dense
        @test Matrix(H_sparse) == H_dense
        @test dense_matrix(Q) == H_dense
        @test Matrix(sparse_matrix(Q)) == H_dense
        @test c_sparse == c_dense
    end

    @testset "Tensor evaluation matches dense H*kron" begin
        Q = QuadraticTensor(
            2,
            3,
            [1, 1, 2],
            [1, 2, 3],
            [1, 3, 2],
            [2.0, -1.5, 0.25],
        )
        H = dense_matrix(Q)
        x = [0.2, -0.5, 1.1]

        out = zeros(2)
        evaluate_quadratic!(out, Q, x)
        @test isapprox(out, H * kron(x, x); atol=1e-12, rtol=1e-12)

        base = [1.0, -2.0]
        expected = copy(base) .+ H * kron(x, x)
        evaluate_quadratic!(base, Q, x; reset=false)
        @test isapprox(base, expected; atol=1e-12, rtol=1e-12)
    end

    @testset "Tensor Galerkin projection matches dense projection" begin
        Q = QuadraticTensor(
            4,
            4,
            [1, 2, 3, 4],
            [1, 1, 2, 3],
            [1, 4, 3, 4],
            [1.25, -0.5, 0.75, -1.0],
        )
        A = reshape(collect(1.0:16.0), 4, 4) ./ 20
        c = [0.1, -0.2, 0.3, -0.4]
        Phi = Matrix(qr([1.0 0.2; 0.3 1.0; -0.5 0.4; 0.1 -0.3]).Q)[:, 1:2]

        A_dense, H_dense, c_dense = galerkin_project(A, dense_matrix(Q), c, Phi)
        A_tensor, Q_hat, c_tensor = galerkin_project(A, Q, c, Phi)

        @test isapprox(A_tensor, A_dense; atol=1e-12, rtol=1e-12)
        @test isapprox(c_tensor, c_dense; atol=1e-12, rtol=1e-12)
        @test isapprox(dense_matrix(Q_hat), H_dense; atol=1e-12, rtol=1e-12)
    end

    @testset "Tensor Galerkin projection handles larger sparse operators" begin
        n = 8
        r = 3
        Q = QuadraticTensor(
            n,
            n,
            [1, 2, 3, 4, 5, 6, 7, 8, 4, 2],
            [1, 1, 2, 3, 4, 5, 6, 7, 2, 1],
            [2, 8, 3, 5, 4, 6, 8, 8, 7, 8],
            [1.2, -0.7, 0.3, 2.1, -1.0, 0.8, -0.4, 0.6, 1.5, -0.2],
        )
        A = reshape(collect(1.0:64.0), n, n) ./ 100
        c = range(-0.2, 0.2; length=n) |> collect
        Phi = Matrix(qr([
            1.0 0.2 -0.3
            0.4 1.0 0.5
            -0.2 0.1 1.0
            0.8 -0.4 0.2
            -0.6 0.7 -0.1
            0.3 -0.8 0.4
            0.5 0.6 -0.7
            -0.1 0.3 0.9
        ]).Q)[:, 1:r]

        A_dense, H_dense, c_dense = galerkin_project(A, dense_matrix(Q), c, Phi)
        A_tensor, Q_hat, c_tensor = galerkin_project(A, Q, c, Phi)

        @test isapprox(A_tensor, A_dense; atol=1e-12, rtol=1e-12)
        @test isapprox(c_tensor, c_dense; atol=1e-12, rtol=1e-12)
        @test isapprox(dense_matrix(Q_hat), H_dense; atol=1e-12, rtol=1e-12)
    end

    @testset "rom_rhs supports tensor operators" begin
        Q = QuadraticTensor(2, 2, [1, 2], [1, 1], [2, 2], [3.0, -2.0])
        A = [0.5 0.1; -0.2 0.3]
        c = [0.05, -0.1]
        x = [0.4, -0.7]

        du_dense = zeros(2)
        du_tensor = zeros(2)
        rom_rhs!(du_dense, x, (A, dense_matrix(Q), c), 0.0)
        rom_rhs!(du_tensor, x, (A, Q, c), 0.0)

        @test isapprox(du_tensor, du_dense; atol=1e-12, rtol=1e-12)
    end

    @testset "rom_rhs supports projected tensor operators" begin
        Q = QuadraticTensor(5, 5, [1, 2, 4, 5], [1, 1, 2, 3], [3, 5, 4, 5], [1.0, -2.0, 0.5, 1.5])
        A = reshape(collect(1.0:25.0), 5, 5) ./ 50
        c = [0.1, -0.2, 0.0, 0.3, -0.1]
        Phi = Matrix(qr([
            1.0 0.2 -0.4
            -0.3 1.0 0.5
            0.7 -0.6 1.0
            0.1 0.8 -0.2
            -0.5 0.4 0.3
        ]).Q)[:, 1:3]

        A_dense, H_dense, c_dense = galerkin_project(A, dense_matrix(Q), c, Phi)
        A_tensor, Q_hat, c_tensor = galerkin_project(A, Q, c, Phi)
        a = [0.2, -0.4, 0.7]

        da_dense = zeros(3)
        da_tensor = zeros(3)
        rom_rhs!(da_dense, a, (A_dense, H_dense, c_dense), 0.0)
        rom_rhs!(da_tensor, a, (A_tensor, Q_hat, c_tensor), 0.0)

        @test isapprox(da_tensor, da_dense; atol=1e-12, rtol=1e-12)
    end
end
