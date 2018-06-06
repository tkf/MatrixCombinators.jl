@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
    const ComplexF64 = Complex128
    const ComplexF32 = Complex64
else
    using Test
    using Random
    import Pkg
end

# Monkey patch for resolving mul!(::Matrix, ::Diagonal, ::Adjoint) ambiguity.
# https://github.com/JuliaLang/julia/pull/27405
@static if VERSION >= v"0.7-"
    import LinearAlgebra
    using LinearAlgebra: Diagonal, Adjoint, Transpose

    let A = Diagonal([100, 200, 300])
        B = reshape(collect(1:9), (3, 3))
        Y = zeros(3, 3)

        need_def = try
            LinearAlgebra.mul!(Y, A, B')
            false
        catch err
            if ! (err isa MethodError)
                rethrow()
            end
            true
        end

        if need_def
            LinearAlgebra.mul!(out::AbstractMatrix, A::Diagonal,
                               in::Adjoint{<:Any, <: AbstractMatrix}) =
                out .= A.diag .* in
            LinearAlgebra.mul!(out::AbstractMatrix, A::Diagonal,
                               in::Transpose{<:Any, <: AbstractMatrix}) =
                out .= A.diag .* in
            LinearAlgebra.mul!(out::AbstractMatrix,
                               A::Adjoint{<:Any, <:Diagonal},
                               in::Adjoint{<:Any, <:AbstractMatrix}) =
                out .= adjoint.(A.parent.diag) .* in
            LinearAlgebra.mul!(out::AbstractMatrix,
                               A::Transpose{<:Any, <:Diagonal},
                               in::Transpose{<:Any, <:AbstractMatrix}) =
                out .= transpose.(A.parent.diag) .* in
        end
    end
end

using MatrixCombinators
using MatrixCombinators: mul!, adjoint, Adjoint, Transpose, empty_array,
    SparseMatrixCSC, sparse
using MatrixCombinators.LinearAlgebra: transpose

eager_t = Dict('N' => identity, 'C' => adjoint, 'T' => transpose)
lazy_t = Dict('N' => identity, 'C' => Adjoint, 'T' => Transpose)

"""
Combination of pairs of N, C, T excluding (C, T) and (T, C).
"""
t_pairs = [(cA, cB) for cA in "NCT", cB in "NCT"
           if cA == 'N' || cB == 'N' || cA == cB]
