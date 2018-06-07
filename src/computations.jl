"""
    _mul!([executor = M.executor], Y, M, X)

Internal interface for `mul!(Y, M, X)` for the case `M` is a
`PairedMatrices` and alike.  It is for *definition* and must not be
called (in most of the cases).

`_mul!` is called via `mul!` in Julia >= 0.7 (see interface10.jl) and
via `A_mul_B!`, `Ac_mul_Bc!`, etc., in Julia 0.6 (see interface06.jl).
"""
function _mul! end

_mul!(Y, M::PairedMatrices, X) = _mul!(M.executor, Y, M, X)


const AdjOrTrOfPair = Union{<: Adjoint{<: Any, <: PairedMatrices},
                            <: Transpose{<: Any, <: PairedMatrices}}
const AdjTrOrPair = Union{PairedMatrices, AdjOrTrOfPair}

# First convert Adjoint/Transpose to a pair and then "execute" it:
_mul!(Y, M::AdjOrTrOfPair, X) = _mul!(Y, do_tr(M), X)


_allocate_mul!(executor, M, X::AbstractVector) =
    allocate!(executor, size(M.B, 1))
_allocate_mul!(executor, M, X::AbstractMatrix) =
    allocate!(executor, (size(M.B, 1), size(X, 2)))


function _mul!(executor::DumbExecutor, Y, M::AddedMatrices, X)
    b_out = _allocate_mul!(executor, M, X)
    mul!(Y, M.A, X)
    mul!(b_out, M.B, X)
    Y .+= b_out
    return Y
end
# Note: This is a default fallback implementation.  If one of the
# matrix A or B defines gemm!-like function, the definition in
# [[./optimizations/gmul.jl]] is used.

# It's more complicated to fallback to the above case when M.A or M.B
# is an UniformScaling than to define the optimized version.  So let's
# define the optimized version here.
function _mul!(executor::DumbExecutor,
               Y,
               M::AddedMatrices{TE, <: UniformScaling},
               X,
               ) where {TE}
    mul!(Y, M.B, X)
    @. Y += M.A.λ * X
    return Y
end

function _mul!(executor::DumbExecutor,
               Y,
               M::AddedMatrices{TE, TA, <: UniformScaling},
               X,
               ) where {TE, TA}
    mul!(Y, M.A, X)
    @. Y += M.B.λ * X
    return Y
end


function _mul!(executor::AllocatingExecutor, Y, M::MultipliedMatrices, X)
    b_out = _allocate_mul!(executor, M, X)
    mul!(b_out, M.B, X)
    mul!(Y, M.A, b_out)
    return Y
end

function _mul!(executor::AllocatingExecutor,
               Y,
               M::MultipliedMatrices{TE, <: UniformScaling},
               X,
               ) where {TE}
    mul!(Y, M.B, X)
    rmul!(Y, M.A.λ)
    return Y
end

function _mul!(executor::AllocatingExecutor,
               Y,
               M::MultipliedMatrices{TE, TA, <: UniformScaling},
               X,
               ) where {TE, TA}
    mul!(Y, M.A, X)
    rmul!(Y, M.B.λ)
    return Y
end

# TODO: improve
out_prototype(M) = M.A
out_prototype(M::PairedMatrices{<: Any, <: UniformScaling}) = M.B


function Base.:*(M::PairedMatrices, x::AbstractVector)
    y = similar(out_prototype(M), size(M, 1))
    mul!(y, M, x)
    return y
end

function Base.:*(M::PairedMatrices, X::AbstractMatrix)
    Y = similar(out_prototype(M), (size(M, 1), size(X, 2)))
    mul!(Y, M, X)
    return Y
end
