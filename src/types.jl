abstract type PairedMatrices{TE, TA, TB, ALC} <: AbstractMatrix{TE} end

struct AddedMatrices{TE, TA, TB, ALC} <: PairedMatrices{TE, TA, TB, ALC}
    A::TA
    B::TB
    allocator::ALC

    function AddedMatrices(A::TA, B::TB,
                           allocator::ALC = allocator_for(A, B),
                           ) where {TA, TB, ALC}
        @assert size(A) == size(B)
        TE = promote_type(eltype(A), eltype(B))
        return new{TE, TA, TB, ALC}(A, B, allocator)
    end
end

const added = AddedMatrices

struct MultipliedMatrices{TE, TA, TB, ALC} <: PairedMatrices{TE, TA, TB, ALC}
    A::TA
    B::TB
    allocator::ALC

    function MultipliedMatrices(A::TA, B::TB,
                                allocator::ALC = allocator_for(A, B),
                                ) where {TA, TB, ALC}
        @assert size(A, 2) == size(B, 1)
        TE = promote_type(eltype(A), eltype(B))
        return new{TE, TA, TB, ALC}(A, B, allocator)
    end
end

const muled = MultipliedMatrices


# --- Utilities

"""
Convert Adjoint/Transpose of a pair to a pair of Adjoint/Transpose
recursively.
"""
do_tr(M::Adjoint{<: Any, <: AddedMatrices}) =
    AddedMatrices(do_tr(Adjoint(M.parent.A)),
                  do_tr(Adjoint(M.parent.B)),
                  M.parent.allocator)

do_tr(M::Transpose{<: Any, <: AddedMatrices}) =
    AddedMatrices(do_tr(Transpose(M.parent.A)),
                  do_tr(Transpose(M.parent.B)),
                  M.parent.allocator)

do_tr(M::Adjoint{<: Any, <: MultipliedMatrices}) =
    MultipliedMatrices(do_tr(Adjoint(M.parent.B)),
                       do_tr(Adjoint(M.parent.A)),
                       M.parent.allocator)

do_tr(M::Transpose{<: Any, <: MultipliedMatrices}) =
    MultipliedMatrices(do_tr(Transpose(M.parent.B)),
                       do_tr(Transpose(M.parent.A)),
                       M.parent.allocator)

do_tr(M) = M


allocate!(M::PairedMatrices, dims) = allocate!(M.allocator, dims)


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

function materialize(T::Type{<: AbstractMatrix}, M::AddedMatrices)
    Y = empty_array(T, size(M))
    @. Y = M.A + M.B
    return Y
end

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

function materialize(T::Type{<: AbstractMatrix}, M::MultipliedMatrices)
    Y = empty_array(T, size(M))
    A_mul_B!(Y, M.A, M.B)
    return Y
end

Base.convert(T::Type{Matrix}, M::PairedMatrices) = materialize(T, M)
Base.convert(T::Type{Array}, M::PairedMatrices) = convert(Matrix, M)
# Those above are similar to how convert(..., ::SparseMatrixCSC) is
# defined in julia/stdlib/SparseArrays/src/sparsematrix.jl.
