include("preamble.jl")

range_mat(n = 3, m = n) = reshape(collect(1:n * m), (n, m))

ab_arrays = let
    A1 = range_mat()

    [
        A1,
        sparse(A1),
        (@view A1[1:end, 1:end]),
    ]
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
         MatrixCombinators.added_ops,
         ab_arrays,
         ab_arrays),
        (MatrixCombinators.muled,
         *,
         MatrixCombinators.muled_ops,
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

                b_out = B * X
                M = combinator(A, B, b_out)

                TE = promote_type(eltype(A), eltype(B), eltype(X))
                desired = Array{TE}((size(A, 1), size(X′, 2)))
                if X isa AbstractVector
                    desired = desired[:, 1]
                end

                actual = similar(desired)
                f(desired, nonlazy(A, B), X′)
                f(actual, M, X′)

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

            # convert
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
