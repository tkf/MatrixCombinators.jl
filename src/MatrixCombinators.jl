module MatrixCombinators

@static if VERSION < v"0.7.0-"
    const LinearAlgebra = Base.LinAlg
else
    using LinearAlgebra
end
const A_mul_B! = LinearAlgebra.A_mul_B!


abstract type Allocator end

# struct NonAllocator <: Allocator end

struct GrowingCacheAllocator{T} <: Allocator
    cache::T
end

GrowingCacheAllocator(V::Type{<: AbstractVector}, len) =
    GrowingCacheAllocator(V(len))
GrowingCacheAllocator(E::Type{<: Number}, len) =
    GrowingCacheAllocator(Vector{E}, len)

function allocate!(allocator::GrowingCacheAllocator, len::Int)
    if length(allocator.cache) < len
        resize!(allocator.cache, len)
    end
    return view(allocator.cache, 1:len)
end

function allocate!(allocator::Allocator, dims::Tuple)
    v = allocate!(allocator, prod(dims))
    return reshape(v, dims)
end

function allocator_for(A, B)
    E = promote_type(eltype.((A, B))...)
    len = size(B, 1)
    return GrowingCacheAllocator(E, len)
end


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


# --- Computations

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

end # module
