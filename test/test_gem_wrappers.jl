include("preamble.jl")

using MatrixCombinators: _gemv!, _gemm!, _amul!,
    adjoint, transpose, Adjoint, Transpose

conc = Dict('N' => identity, 'C' => adjoint, 'T' => transpose)
lazy = Dict('N' => identity, 'C' => Adjoint, 'T' => Transpose)

@testset "gemv!" begin
    for (seed, (n, m)) in enumerate([(3, 3), (3, 5), (11, 7)])
        rng = MersenneTwister(seed)
        A = randn(rng, (n, m))
        x = randn(rng, m)
        y = randn(rng, n)
        α = randn(rng)
        β = randn(rng)
        for c in "NCT"
            t = conc[c]
            A_ = Matrix(t(A))
            gemv_desired = α .* (t(A_) * x) .+ β .* y
            gemv_actual = copy(y)
            _gemv!(c, α, A_, x, β, gemv_actual)
            @test gemv_actual ≈ gemv_desired

            amul_actual = t(A_) * x .+ y
            amul_desired = copy(y)
            _amul!(amul_desired, lazy[c](A_), x)
            @test amul_actual ≈ amul_desired
        end
    end
end

@testset "gemm!" begin
    for (seed, (n, m, k)) in enumerate([(3, 3, 2),
                                        (3, 5, 4),
                                        (11, 7, 8)])
        rng = MersenneTwister(seed)
        A = randn(rng, (n, m))
        B = randn(rng, (m, k))
        C = randn(rng, (n, k))
        α = randn(rng)
        β = randn(rng)
        for cA in "NCT", cB in "NCT"
            tA = conc[cA]
            tB = conc[cB]
            if cA != 'N' && cA != cB
                continue
            end
            A_ = Matrix(tA(A))
            B_ = Matrix(tB(B))
            gemm_desired = α .* (tA(A_) * tB(B_)) .+ β .* C
            gemm_actual = copy(C)
            _gemm!(cA, cB, α, A_, B_, β, gemm_actual)
            @test gemm_actual ≈ gemm_desired

            amul_actual = tA(A_) * tB(B_) .+ C
            amul_desired = copy(C)
            _amul!(amul_desired, lazy[cA](A_), lazy[cB](B_))
            @test amul_actual ≈ amul_desired
        end
    end
end
