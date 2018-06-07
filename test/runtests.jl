include("preamble.jl")

@testset "$file" for file in [
        "test_gem_wrappers.jl",
        "test_builders.jl",
        "test_core.jl",
        "test_readme.jl",
        ]
    @time include(file)
end
