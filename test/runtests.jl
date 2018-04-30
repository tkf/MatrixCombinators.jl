include("preamble.jl")

@testset "$file" for file in [
        "test_core.jl",
        "test_readme.jl",
        ]
    @time include(file)
end
