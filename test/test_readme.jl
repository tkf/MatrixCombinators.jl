let in_code = false, code = []
    for line in readlines(joinpath(@__DIR__, "..", "README.md"))
        if line == "```julia"
            in_code = true
        elseif line == "```"
            in_code = false
        elseif in_code
            push!(code, line)
        end
    end
    write(joinpath(@__DIR__, "README.jl"), join(code, "\n"))
end

include("preamble.jl")
using MatrixCombinators.LinearAlgebra: A_mul_B!
include("README.jl")
