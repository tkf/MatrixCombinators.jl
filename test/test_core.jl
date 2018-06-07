include("preamble.jl")


range_mat(n = 3, m = n) = reshape(collect(1:n * m), (n, m))

array_variants(A1) =
    [
        A1,
        sparse(A1),
        (@view A1[1:end, 1:end]),
    ]

a_arrays = let
    A1 = range_mat()

    vcat(
        array_variants(A1),          # Int arrays
        array_variants(A1 .* 1.0),   # Float64 arrays
    )
end

ab_arrays_default = [(A, B) for A in a_arrays, B in a_arrays][:]
ab_arrays_default = vcat(ab_arrays_default, [
    (A, I) for A in a_arrays
], [
    (I, B) for B in a_arrays
])
ab_arrays_added = ab_arrays_default
ab_arrays_muled = ab_arrays_default


@testset "$combinator" for (combinator,
                            nonlazy,
                            ab_arrays,
                            ) in
    [
        (MatrixCombinators.added,
         +,
         ab_arrays_added,
         ),
        (MatrixCombinators.muled,
         *,
         ab_arrays_muled,
         ),
    ]

    @testset "cA=$cA, cB=$cB" for (cA, cB) in t_pairs
        f = (Y, A, B) -> mul!(Y, eager_t[cA](A), eager_t[cB](B))
        for (A, B) in ab_arrays

            m, n = size(combinator(A, B))
            x_arrays = [
                collect(1:n),
                reshape(collect(1:n^2), (n, n)),
            ]
            # TODO: non-square matrix

            for X in x_arrays
                X′ = eager_t[cB](X)
                if X isa AbstractVector && X !== X′
                    continue
                end
                X′ = X′ .+ 0  # materialize

                M = combinator(A, B)

                TE = promote_type(eltype(A), eltype(B), eltype(X))
                desired = empty_array(Array{TE}, (m, size(X′, 2)))
                if X isa AbstractVector
                    desired = desired[:, 1]
                end

                actual = similar(desired)
                f(desired, nonlazy(A, B), X′)
                f(actual, M, X′)

                @test actual ≈ desired

                if ! (X isa AbstractVector)
                    if cB in "TC"
                        continue
                    end

                    v = X′[:, 1]

                    desired = empty_array(Array{TE}, m)
                    actual = similar(desired)

                    f(desired, nonlazy(A, B), v)
                    f(actual, M, v)

                    @test actual ≈ desired
                end
            end
        end
    end

    @testset "*" begin
        for (A, B) in ab_arrays_default

            _, n = size(combinator(A, B))
            x_arrays = [
                collect(1:n),
                reshape(collect(1:n^2), (n, n)),
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
        for (A, B) in ab_arrays_default

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

        for TA in [Int, Float64, ComplexF64],
            TB in [Int, Float64, ComplexF64]

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
         ) in [
             (MatrixCombinators.AddedMatrices,
              +,
              ),
             (MatrixCombinators.MultipliedMatrices,
              *,
              ),
         ],
         dest in [
             Array,
             Array{Float64},
             Matrix,
             Matrix{Float64},
             SparseMatrixCSC,
             SparseMatrixCSC{Float64},
         ]

    for (A, B) in ab_arrays_default

        M = combinator(A, B)
        D = nonlazy(A, B)
        C = MatrixCombinators.materialize(dest, M)
        @test C ≈ D
    end
end
