const DiagTimesCSC{
    TE,
    TA <: Diagonal,
    TB <: SparseMatrixCSC,
    EXR,
} = MultipliedMatrices{TE, TA, TB, EXR}

"""
    gmul!(Y, M::DiagTimesCSC, X, α, β)

An optimized implementation of the case `M` is a `Diagonal` matrix
right-multiplied by a `SparseMatrixCSC`:

    Y = α * D * S * X + β * Y
            `---'
              M
"""
function gmul!(Y::AbstractMatrix, M::DiagTimesCSC, X::AbstractMatrix,
               alpha, beta)
    @assert (size(Y, 1), size(X, 1)) == size(M.A) == size(M.B)
    @assert size(Y, 2) == size(X, 2)
    if beta != 1
        scale!(Y, beta)
    end
    @inbounds @views for i in 1:size(Y, 2)
        _gmul_diag_csc!(Y[:, i], M.A.diag, M.B, X[:, i], alpha)
    end
    return Y
end

function gmul!(y::AbstractVector, M::DiagTimesCSC, x::AbstractVector,
               alpha, beta)
    @assert (size(y, 1), size(x, 1)) == size(M.A) == size(M.B)
    if beta != 1
        scale!(y, beta)
    end
    return _gmul_diag_csc!(y, M.A.diag, M.B, x, alpha)
end

@inline function _gmul_diag_csc!(y, diag, S, x, alpha)
    rows = rowvals(S)
    vals = nonzeros(S)
    @inbounds for j in 1:size(x, 1)
        # TODO: maybe use MuladdMacro.jl
        @fastmath @simd for k in nzrange(S, j)
            i = rows[k]
            Sij = vals[k]
            y[i] += alpha * diag[i] * Sij * x[j]
        end
    end
end

function _mul!(::DefaultExecutor, Y, M::DiagTimesCSC, X)
    fill!(Y, zero(eltype(Y)))
    amul!(Y, M, X)
    return Y
end

_mul!(executor::DefaultExecutor, Y, M::DiagTimesCSC,
      X::Union{<: Adjoint, <: Transpose}) =
    _mul!(DumbExecutor(executor.allocator), Y, M, X)
# Fallback to the non-optimized implementation.
