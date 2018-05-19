abstract type Allocator end

# struct NonAllocator <: Allocator end

struct GrowingCacheAllocator{T} <: Allocator
    cache::T
end

GrowingCacheAllocator(V::Type{<: AbstractVector}, len) =
    GrowingCacheAllocator(empty_array(V, len))
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
