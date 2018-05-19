@static if VERSION >= v"0.7.0-"
    const BLAS = LinearAlgebra.BLAS
end


function _gemm!(tA, tB,
                alpha::T,
                A::AbstractVecOrMat{T},
                B::AbstractVecOrMat{T},
                beta::T,
                C::AbstractVecOrMat{T},
                ) where {T <: LinearAlgebra.BlasFloat}

    # Taken from LinearAlgebra.gemm_wrapper!:
    mA, nA = LinearAlgebra.lapack_size(tA, A)
    mB, nB = LinearAlgebra.lapack_size(tB, B)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
end

_gemv!(tA, alpha, A, X, beta, Y) = BLAS.gemv!(tA, alpha, A, X, beta, Y)


function _gemm!(tA, tB,
                alpha::Number,
                A::SparseMatrixCSC,
                B::StridedVecOrMat,
                beta::Number,
                C::StridedVecOrMat)
    @assert tA in ('N', 'C', 'T')
    @assert tB == 'N'
    if tA == 'N'
        A_mul_B!(alpha, A, B, beta, C)
    elseif tA == 'C'
        Ac_mul_B!(alpha, A, B, beta, C)
    elseif tA == 'T'
        At_mul_B!(alpha, A, B, beta, C)
    end
end

function _gemv!(tA,
                alpha::Number,
                A::SparseMatrixCSC,
                X::StridedVector,
                beta::Number,
                Y::StridedVector)
    _gemm!(tA, 'N', alpha, A, X, beta, Y)
end


function _amul!(Y::AbstractMatrix{TY},
                A::AbstractMatrix{TA},
                B::AbstractMatrix{TB},
                ) where {TY, TA, TB}
    et = promote_type(TY, TA, TB)
    tA = tchar(A)
    tB = tchar(B)
    _gemm!(tA, tB, one(et), peel(A), peel(B), one(et), Y)
end

function _amul!(Y::AbstractVector{TY},
                A::AbstractMatrix{TA},
                B::AbstractVector{TB},
                ) where {TY, TA, TB}
    et = promote_type(TY, TA, TB)
    tA = tchar(A)
    _gemv!(tA, one(et), peel(A), B, one(et), Y)
end
