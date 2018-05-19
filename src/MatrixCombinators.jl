module MatrixCombinators

include("utils.jl")
include("gem_wrappers.jl")
include("allocators.jl")
include("executors.jl")
include("types.jl")
include("computations.jl")

@static if VERSION < v"0.7.0-"
    include("interface06.jl")
else
    include("interface10.jl")
end

end # module
