abstract type PairedMatrices{TA, TB, ALC} end

struct AddedMatrices{TA, TB, ALC} <: PairedMatrices{TA, TB, ALC}
    A::TA
    B::TB
    allocator::ALC

    function AddedMatrices(A::TA, B::TB,
                           allocator::ALC = allocator_for(A, B),
                           ) where {TA, TB, ALC}
        @assert size(A) == size(B)
        return new{TA, TB, ALC}(A, B, allocator)
    end
end

const added = AddedMatrices

struct MultipliedMatrices{TA, TB, ALC} <: PairedMatrices{TA, TB, ALC}
    A::TA
    B::TB
    allocator::ALC

    function MultipliedMatrices(A::TA, B::TB,
                                allocator::ALC = allocator_for(A, B),
                                ) where {TA, TB, ALC}
        @assert size(A, 2) == size(B, 1)
        return new{TA, TB, ALC}(A, B, allocator)
    end
end

const muled = MultipliedMatrices


# --- Utilities

allocate!(M::PairedMatrices, dims) = allocate!(M.allocator, dims)

empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(dims)
empty_array(::Type{<: SparseMatrixCSC{T}}, dims,) where {T} =
    spzeros(T, dims...)


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

Base.IndexStyle(::Type{<:AddedMatrices{TA, TB}}) where {TA, TB} =
    preferred_style(IndexStyle(TA), IndexStyle(TB))

function Base.convert(T::Type{<: AbstractArray}, M::AddedMatrices)
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
    return (M * x)[i]
end

Base.IndexStyle(::Type{<:MultipliedMatrices}) = IndexCartesian()

function Base.convert(T::Type{<: AbstractArray}, M::MultipliedMatrices)
    Y = empty_array(T, size(M))
    A_mul_B!(Y, M.A, M.B)
    return Y
end
