"""
`Executor` type is used as one of the type parameter of
`PairedMatrices` and determines the execution strategy of the given
computation graph.

Users can define their own `Executor` subtype in order to define their
optimized computation for any (sub) graph.
"""
abstract type Executor end

struct AllocatingExecutor{mode, ALC <: Allocator}
    allocator::ALC
end

AllocatingExecutor{mode}(allocator::ALC) where {mode, ALC <: Allocator} =
    AllocatingExecutor{mode, ALC}(allocator)

const DefaultExecutor = AllocatingExecutor{:default}
const DumbExecutor = AllocatingExecutor{:dumb}

executor_for(A, B) =
    DefaultExecutor(allocator_for(A, B))

allocate!(E::AllocatingExecutor, dims) = allocate!(E.allocator, dims)
