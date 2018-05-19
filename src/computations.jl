"""
Internal interface for `mul!(Y, M, X)` for the case `M` is a
`PairedMatrices` and alike.  It is for *definition* and must not be
called (in most of the cases).
"""
function _mul! end

const AdjOrTrOfPair = Union{<: Adjoint{<: Any, <: PairedMatrices},
                            <: Transpose{<: Any, <: PairedMatrices}}

# First convert Adjoint/Transpose to a pair and then "execute" it:
_mul!(Y, M::AdjOrTrOfPair, X) = _mul!(Y, do_tr(M), X)

_allocate_mul!(M, X::AbstractVector) = allocate!(M, size(M.B, 1))
_allocate_mul!(M, X::AbstractMatrix) = allocate!(M, (size(M.B, 1), size(X, 2)))


function _mul!(Y, M::AddedMatrices, X)
    b_out = _allocate_mul!(M, X)
    mul!(Y, M.A, X)
    mul!(b_out, M.B, X)
    Y .+= b_out
    return Y
end
# Note: using gemm! would be better since I can get rid of b_out here.
# But it seems there is no consistent gemm! interface for various
# structured/sparse matrix.  I need to define a wrapper first for this
# optimization.


function _mul!(Y, M::MultipliedMatrices, X)
    b_out = _allocate_mul!(M, X)
    mul!(b_out, M.B, X)
    mul!(Y, M.A, b_out)
    return Y
end


function Base.:*(M::PairedMatrices, x::AbstractVector)
    y = similar(@view M.A[:, 1])
    mul!(y, M, x)
    return y
end

function Base.:*(M::PairedMatrices, X::AbstractMatrix)
    Y = similar(M.A, (size(M, 1), size(X, 2)))  # TODO: improve
    mul!(Y, M, X)
    return Y
end
