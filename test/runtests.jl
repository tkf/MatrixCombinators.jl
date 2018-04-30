using MatrixCombinators
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

ab_arrays = let
    A1 = reshape(collect(1:9), (3, 3))

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
        f = eval(name)
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
                if f in (Base.A_mul_Bt!, Base.At_mul_Bt!)
                    X′ = transpose(X)
                elseif f in (Base.A_mul_Bc!, Base.Ac_mul_Bc!)
                    X′ = X'
                else
                    X′ = X
                end
                b_out = B * X
                M = combinator(A, B, b_out)

                desired = nonlazy(A, B) * X
                actual = similar(desired)
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
            for i in 1:size(M, 1), j in 1:size(M, 2)
                @test M[i, j] ≈ D[i, j]
            end

            # convert
            C = convert(Array, M)
            @test C ≈ D

            if ! (D isa Array)
                C = convert(typeof(D), M)
                @test C ≈ D
            end
        end
    end
end
