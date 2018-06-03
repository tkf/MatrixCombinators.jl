include("preamble.jl")

added_ops = :(
    LinearAlgebra.A_mul_B!,
    LinearAlgebra.A_mul_Bt!, LinearAlgebra.At_mul_B!, LinearAlgebra.At_mul_Bt!,
    LinearAlgebra.A_mul_Bc!, LinearAlgebra.Ac_mul_B!, LinearAlgebra.Ac_mul_Bc!,
).args

muled_ops_nt = :(
    LinearAlgebra.A_mul_B!,
    LinearAlgebra.A_mul_Bt!,
    LinearAlgebra.A_mul_Bc!,
).args

muled_ops_tr = :(
    LinearAlgebra.At_mul_B!, LinearAlgebra.At_mul_Bt!,
    LinearAlgebra.Ac_mul_B!, LinearAlgebra.Ac_mul_Bc!,
).args

muled_ops = (muled_ops_nt..., muled_ops_tr...)


range_mat(n = 3, m = n) = reshape(collect(1:n * m), (n, m))

array_variants(A1) =
    [
        A1,
        sparse(A1),
        (@view A1[1:end, 1:end]),
    ]

ab_arrays = let
    A1 = range_mat()

    vcat(
        array_variants(A1),          # Int arrays
        array_variants(A1 .* 1.0),   # Float64 arrays
    )
end


@testset "$combinator" for (combinator,
                            nonlazy,
                            ops,
                            a_arrays,
                            b_arrays,
                            ) in
    [
        (MatrixCombinators.added,
         +,
         added_ops,
         ab_arrays,
         ab_arrays),
        (MatrixCombinators.muled,
         *,
         muled_ops,
         ab_arrays,
         ab_arrays),
    ]

    @testset "$name" for name in ops
        f = eval(MatrixCombinators, name)
        for A in a_arrays,
            B in b_arrays

            x_arrays = [
                collect(1:size(B, 2)),
                let N = size(B, 2)
                    reshape(collect(1:N^2), (N, N))
                end,
            ]
            # TODO: non-square matrix

            for X in x_arrays
                if f in (A_mul_Bt!, At_mul_Bt!)
                    X′ = transpose(X)
                elseif f in (A_mul_Bc!, Ac_mul_Bc!)
                    X′ = X'
                else
                    X′ = X
                end
                if X isa AbstractVector && X !== X′
                    continue
                end
                X′ = X′ .+ 0  # materialize

                M = combinator(A, B)

                TE = promote_type(eltype(A), eltype(B), eltype(X))
                desired = Array{TE}((size(A, 1), size(X′, 2)))
                if X isa AbstractVector
                    desired = desired[:, 1]
                end

                actual = similar(desired)
                f(desired, nonlazy(A, B), X′)
                f(actual, M, X′)

                @test actual ≈ desired

                if ! (X isa AbstractVector)
                    if f in (A_mul_Bt!, At_mul_Bt!, A_mul_Bc!, Ac_mul_Bc!)
                        continue
                    end

                    v = X′[:, 1]

                    desired = Array{TE}(size(A, 1))
                    actual = similar(desired)

                    f(desired, nonlazy(A, B), v)
                    f(actual, M, v)

                    @test actual ≈ desired
                end
            end
        end
    end

    @testset "*" begin
        for A in a_arrays,
            B in b_arrays

            x_arrays = [
                collect(1:size(B, 2)),
                let N = size(B, 2)
                    reshape(collect(1:N^2), (N, N))
                end,
            ]
            # TODO: non-square matrix

            D = nonlazy(A, B)
            for X in x_arrays
                M = combinator(A, B)

                actual = M * X
                desired = D * X
                @test actual ≈ desired
            end
        end
    end

    @testset "interface" begin
        for A in a_arrays,
            B in b_arrays

            M = combinator(A, B)
            D = nonlazy(A, B)

            # getindex
            diff_at = []
            for i in 1:size(M, 1), j in 1:size(M, 2)
                if ! (M[i, j] ≈ D[i, j])
                    push!(diff_at, (i, j))
                end
            end
            @test diff_at == []

            if IndexStyle(typeof(M)) isa IndexLinear
                diff_at = []
                for i in 1:length(M)
                    if ! (M[i] ≈ D[i])
                        push!(diff_at, i)
                    end
                end
                @test diff_at == []
            end

            # convert
            C = convert(Array, M)
            @test C ≈ D

            TE = promote_type(eltype(A), eltype(B))
            C = convert(Array{TE}, M)
            @test C ≈ D

            if ! (D isa Array)
                C = convert(typeof(D), M)
                @test C ≈ D
            end
        end

        for TA in [Int, Float64, Complex128],
            TB in [Int, Float64, Complex128]

            A = Array{TA}(range_mat())
            B = Array{TB}(range_mat())
            M = combinator(A, B)
            @assert eltype(M) == promote_type(TA, TB)
        end
    end
end


@testset "materialize($dest, ::$combinator)" for
        (combinator,
         nonlazy,
         a_arrays,
         b_arrays,
         ) in [
             (MatrixCombinators.AddedMatrices,
              +,
              ab_arrays,
              ab_arrays),
             (MatrixCombinators.MultipliedMatrices,
              *,
              ab_arrays,
              ab_arrays),
         ],
         dest in [
             Array,
             Array{Float64},
             Matrix,
             Matrix{Float64},
             SparseMatrixCSC,
             SparseMatrixCSC{Float64},
         ]

    for A in a_arrays,
        B in b_arrays

        M = combinator(A, B)
        D = nonlazy(A, B)
        C = MatrixCombinators.materialize(dest, M)
        @test C ≈ D
    end
end
