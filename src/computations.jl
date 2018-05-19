# AddedMatrices
added_ops = :(
    LinearAlgebra.A_mul_B!,
    LinearAlgebra.A_mul_Bt!, LinearAlgebra.At_mul_B!, LinearAlgebra.At_mul_Bt!,
    LinearAlgebra.A_mul_Bc!, LinearAlgebra.Ac_mul_B!, LinearAlgebra.Ac_mul_Bc!,
).args
for TY in [AbstractMatrix, AbstractVector]
    for f in added_ops
        if TY === AbstractVector
            b_out = :(allocate!(M, size(M.B, 1)))
        elseif eval(f) in (LinearAlgebra.At_mul_B!,
                           LinearAlgebra.Ac_mul_B!)
            b_out = :(allocate!(M, (size(M.B, 2), size(X, 2))))
        elseif eval(f) in (LinearAlgebra.At_mul_Bt!,
                           LinearAlgebra.Ac_mul_Bc!)
            b_out = :(allocate!(M, (size(M.B, 2), size(X, 1))))
        elseif eval(f) in (LinearAlgebra.A_mul_Bt!,
                           LinearAlgebra.A_mul_Bc!)
            b_out = :(allocate!(M, (size(M.B, 1), size(X, 1))))
        else
            b_out = :(allocate!(M, (size(M.B, 1), size(X, 2))))
        end
        @eval function $f(Y::$TY, M::AddedMatrices, X)
            b_out = $b_out
            $f(Y, M.A, X)
            $f(b_out, M.B, X)
            Y .+= b_out
            return Y
        end
    end
end
# Note: using gemm! would be better since I can get rid of b_out here.
# But it seems there is no consistent gemm! interface for various
# structured/sparse matrix.  I need to define a wrapper first for this
# optimization.


# MultipliedMatrices
muled_ops_nt = :(
    LinearAlgebra.A_mul_B!,
    LinearAlgebra.A_mul_Bt!,
    LinearAlgebra.A_mul_Bc!,
).args
muled_ops_tr = :(
    LinearAlgebra.At_mul_B!, LinearAlgebra.At_mul_Bt!,
    LinearAlgebra.Ac_mul_B!, LinearAlgebra.Ac_mul_Bc!,
).args
for TY in [AbstractMatrix, AbstractVector]
    for f in muled_ops_nt
        if TY === AbstractVector
            b_out = :(allocate!(M, size(M.B, 1)))
        else
            b_out = :(allocate!(M, (size(M.B, 1), size(X, 2))))
        end
        @eval function $f(Y::$TY, M::MultipliedMatrices, X)
            b_out = $b_out
            $f(b_out, M.B, X)
            A_mul_B!(Y, M.A, b_out)
            return Y
        end
    end
    for f in muled_ops_tr
        g = Dict(
            LinearAlgebra.At_mul_B!  => LinearAlgebra.At_mul_B!,
            LinearAlgebra.At_mul_Bt! => LinearAlgebra.At_mul_B!,
            LinearAlgebra.Ac_mul_B!  => LinearAlgebra.Ac_mul_B!,
            LinearAlgebra.Ac_mul_Bc! => LinearAlgebra.Ac_mul_B!,
        )[eval(f)]
        if TY === AbstractVector
            a_out = :(allocate!(M, size(M.A, 2)))
        elseif eval(f) in (LinearAlgebra.At_mul_B!,
                           LinearAlgebra.Ac_mul_B!)
            a_out = :(allocate!(M, (size(M.A, 2), size(X, 2))))
        elseif eval(f) in (LinearAlgebra.At_mul_Bt!,
                           LinearAlgebra.Ac_mul_Bc!)
            a_out = :(allocate!(M, (size(M.A, 2), size(X, 1))))
        end
        @eval function $f(Y::$TY, M::MultipliedMatrices, X)
            a_out = $a_out
            $f(a_out, M.A, X)
            $g(Y, M.B, a_out)
            return Y
        end
    end
end
muled_ops = (muled_ops_nt..., muled_ops_tr...)

function Base.:*(M::PairedMatrices, x::AbstractVector)
    y = similar(@view M.A[:, 1])
    A_mul_B!(y, M, x)
    return y
end

function Base.:*(M::PairedMatrices, X::AbstractMatrix)
    Y = similar(M.A, (size(M, 1), size(X, 2)))  # TODO: improve
    A_mul_B!(Y, M, X)
    return Y
end
