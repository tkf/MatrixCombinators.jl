abstract type Allocator end

# struct NonAllocator <: Allocator end

# TODO: What to do element type cannot be determined?
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

    # Heuristics for guessing a nice length to initially allocate:
    if has_size(B)
        len = size(B, 1)
    elseif has_size(A)
        len = size(A, 1)
    else
        error("Both A and B does not have size.")
    end

    return GrowingCacheAllocator(E, len)
end
