include("preamble.jl")

using MatrixCombinators: _gemv!, _gemm!, gmul!, has_gemv, has_gemm, has_gmul,
    adjoint, transpose, Adjoint, Transpose, empty_array

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
            @test has_gemv(typeof.((A_, x, gemv_actual))...)
            _gemv!(c, α, A_, x, β, gemv_actual)
            @test gemv_actual ≈ gemv_desired

            gmul_actual = t(A_) * x .+ y
            gmul_desired = copy(y)
            gmul!(gmul_desired, lazy[c](A_), x)
            @test gmul_actual ≈ gmul_desired
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
            @test has_gemm(typeof.((A_, B_, gemm_actual))...)
            @test gemm_actual ≈ gemm_desired

            gmul_actual = tA(A_) * tB(B_) .+ C
            gmul_desired = copy(C)
            gmul!(gmul_desired, lazy[cA](A_), lazy[cB](B_))
            @test gmul_actual ≈ gmul_desired
        end
    end
end


some_array(T::Type{<: AbstractMatrix}) = empty_array(T, (2, 2))
some_array(T::Type{<: AbstractVector}) = empty_array(T, (2,))


@testset "has_gmul" begin
    abc_types_with_blas_gemm = [
        (:Matrix, :Matrix, :Matrix),
        (:Vector, :Matrix, :Vector),
    ]
    # TODO: Add BlockBandedMatrices.BlockBandedBlock

    abc_types_with_nonblas_gemm = [
        (:Matrix, :SparseMatrixCSC, :Matrix),
        (:Vector, :SparseMatrixCSC, :Vector),
    ]

    abc_types_with_gmul = vcat(
        abc_types_with_blas_gemm,
        abc_types_with_nonblas_gemm,
    )

    @testset "no element type, TA=$TA, TB=$TB, TC=$TC" for
            (TA, TB, TC) in abc_types_with_blas_gemm
        # Should I use `abc_types_with_gmul`?

        TA = eval(TA)
        TB = eval(TB)
        TC = eval(TC)

        @test ! has_gmul(TA, TB, TC)
    end

    for (element_types, abc_types, yes) in [
            ((Float64, Float32, Complex128, Complex64),
             abc_types_with_gmul,
             true),
            ((Int, BigFloat, BigInt),
             abc_types_with_blas_gemm,
             false),
            ((Int, BigFloat, BigInt),
             abc_types_with_nonblas_gemm,
             true),
        ]

        @testset "elty=$elty, TA=$TA, TB=$TB, TC=$TC" for
                elty in element_types,
                (TA, TB, TC) in abc_types

            TA = eval(TA)
            TB = eval(TB)
            TC = eval(TC)

            @test has_gmul(TA{elty}, TB{elty}, TC{elty}) == yes

            A = some_array(TA{elty})
            B = some_array(TB{elty})
            C = some_array(TC{elty})
            @test has_gmul(A, B, C) == yes
        end
    end
end
