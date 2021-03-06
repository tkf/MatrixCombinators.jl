@static if VERSION < v"0.7.0-"
    const LinearAlgebra = Base.LinAlg
    include("compat06.jl")
    import Base: ctranspose, transpose
    const adjoint = ctranspose
else
    using LinearAlgebra
    using SparseArrays
    import Base: adjoint, transpose
end
const A_mul_B! = LinearAlgebra.A_mul_B!


@static if VERSION < v"0.7.0-"
    empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(dims)
else
    empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(undef, dims)
end
empty_array(::Type{<: SparseMatrixCSC{T}}, dims) where {T} =
    spzeros(T, dims...)


has_size(A) = length(methods(size, (typeof(A),))) > 0


peel(A::Union{Adjoint, Transpose}) = A.parent
peel(A) = A


tchar(::Type{<: AbstractMatrix}) = 'N'
tchar(::Type{<: Adjoint}) = 'C'
tchar(::Type{<: Transpose}) = 'T'
tchar(::T) where T = tchar(T)
