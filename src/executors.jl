"""
`Executor` type is used as one of the type parameter of
`PairedMatrices` and determines the execution strategy of the given
computation graph.

Users can define their own `Executor` subtype in order to define their
optimized computation for any (sub) graph.
"""
abstract type Executor end

"""
The default executor type.
"""
struct AllocatingExecutor{ALC <: Allocator}
    allocator::ALC
end

executor_for(A, B) =
    AllocatingExecutor(allocator_for(A, B))

allocate!(E::AllocatingExecutor, dims) = allocate!(E.allocator, dims)
