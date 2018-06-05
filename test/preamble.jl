@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
    const ComplexF64 = Complex128
    const ComplexF32 = Complex64
else
    using Test
    using Random
    import Pkg
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
