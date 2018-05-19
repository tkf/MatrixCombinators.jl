@static if VERSION < v"0.7.0-"
    const LinearAlgebra = Base.LinAlg
    include("compat06.jl")
else
    using LinearAlgebra
    using SparseArrays
end
const A_mul_B! = LinearAlgebra.A_mul_B!


@static if VERSION < v"0.7.0-"
    empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(dims)
else
    empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(undef, dims)
end
empty_array(::Type{<: SparseMatrixCSC{T}}, dims) where {T} =
    spzeros(T, dims...)
