abstract type PairedMatrices{TE, TA, TB, EXR} <: AbstractMatrix{TE} end

struct AddedMatrices{TE, TA, TB, EXR} <: PairedMatrices{TE, TA, TB, EXR}
    A::TA
    B::TB
    executor::EXR

    function AddedMatrices(A::TA, B::TB,
                           executor::EXR = executor_for(A, B),
                           ) where {TA, TB, EXR}
        @assert size(A) == size(B)
        TE = promote_type(eltype(A), eltype(B))
        return new{TE, TA, TB, EXR}(A, B, executor)
    end
end

const added = AddedMatrices

struct MultipliedMatrices{TE, TA, TB, EXR} <: PairedMatrices{TE, TA, TB, EXR}
    A::TA
    B::TB
    executor::EXR

    function MultipliedMatrices(A::TA, B::TB,
                                executor::EXR = executor_for(A, B),
                                ) where {TA, TB, EXR}
        @assert size(A, 2) == size(B, 1)
        TE = promote_type(eltype(A), eltype(B))
        return new{TE, TA, TB, EXR}(A, B, executor)
    end
end

const muled = MultipliedMatrices


# --- Utilities

adjoint(M::PairedMatrices) = do_tr(Adjoint(M))
transpose(M::PairedMatrices) = do_tr(Transpose(M))

"""
Convert Adjoint/Transpose of a pair to a pair of Adjoint/Transpose
recursively.
"""
do_tr(M::Adjoint{<: Any, <: AddedMatrices}) =
    AddedMatrices(do_tr(Adjoint(M.parent.A)),
                  do_tr(Adjoint(M.parent.B)),
                  M.parent.executor)

do_tr(M::Transpose{<: Any, <: AddedMatrices}) =
    AddedMatrices(do_tr(Transpose(M.parent.A)),
                  do_tr(Transpose(M.parent.B)),
                  M.parent.executor)

do_tr(M::Adjoint{<: Any, <: MultipliedMatrices}) =
    MultipliedMatrices(do_tr(Adjoint(M.parent.B)),
                       do_tr(Adjoint(M.parent.A)),
                       M.parent.executor)

do_tr(M::Transpose{<: Any, <: MultipliedMatrices}) =
    MultipliedMatrices(do_tr(Transpose(M.parent.B)),
                       do_tr(Transpose(M.parent.A)),
                       M.parent.executor)

do_tr(M) = M


allocate!(M::PairedMatrices, dims) = allocate!(M.executor, dims)


# --- Array Interface
# https://docs.julialang.org/en/stable/manual/interfaces/#man-interface-array-1

Base.eltype(M::PairedMatrices) = promote_type(eltype(M.A),
                                              eltype(M.B))
Base.length(M::PairedMatrices) = prod(size(M))

Base.size(M::AddedMatrices, dim...) = size(M.A, dim...)
Base.getindex(M::AddedMatrices, i::Int) = M.A[i] + M.B[i]
Base.getindex(M::AddedMatrices, I::Vararg{Int, N}) where N =
    getindex(M.A, I...) +
    getindex(M.B, I...)

preferred_style(t::T, ::T) where {T <: IndexStyle} = t
preferred_style(::IndexStyle, ::IndexStyle) = IndexCartesian()

Base.IndexStyle(::Type{<:AddedMatrices{<: Any, TA, TB}}) where {TA, TB} =
    preferred_style(IndexStyle(TA), IndexStyle(TB))

Base.size(M::MultipliedMatrices) = (size(M.A, 1), size(M.B, 2))

function Base.size(M::MultipliedMatrices, i::Integer)
    if i == 1
        size(M.A, 1)
    else
        size(M.B, i)  # if i > 2, this returns 1
    end
end

function Base.getindex(M::MultipliedMatrices, i::Int, j::Int)
    x = spzeros(Int, size(M, 2))
    x[j] = 1
    y = empty_array(Array{eltype(M)}, size(M, 1))
    _mul!(y, M, x)
    return y[i]
end

Base.IndexStyle(::Type{<:MultipliedMatrices}) = IndexCartesian()


"""
    materialize(T::Type, M::PairedMatrices)

Note: prefer `convert` rather than calling this function directly.

Do the computation of `M` and convert the result into a regular array
of type `T`.

If array type `T` specify the element type, it is used.  If not, the
element type of `M` is used.
"""
materialize(T::Type{<: AbstractArray{<: Number}},
            M::PairedMatrices) =
    _materialize(T, M)

# If Matrix type `TA` does not have "concrete" element type, use the
# element type of `M`:
materialize(TA::Type{<: AbstractArray},
            M::PairedMatrices{TE},
            ) where {TE} =
    materialize(TA{TE}, M)

# Actual implementation of `materialize`.  This is done in a separate
# function so that I don't need to repeat the function body for the
# disambiguation juggling above.
function _materialize(T, M::AddedMatrices)
    Y = empty_array(T, size(M))
    @. Y = M.A + M.B
    return Y
end

function _materialize(T, M::MultipliedMatrices)
    Y = empty_array(T, size(M))
    mul!(Y, M.A, M.B)
    return Y
end


Base.convert(T::Type{Matrix}, M::PairedMatrices) = materialize(T, M)
Base.convert(T::Type{Array}, M::PairedMatrices) = convert(Matrix, M)
# Those above are similar to how convert(..., ::SparseMatrixCSC) is
# defined in julia/stdlib/SparseArrays/src/sparsematrix.jl.
