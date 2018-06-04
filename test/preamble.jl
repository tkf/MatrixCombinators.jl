@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using MatrixCombinators
using MatrixCombinators: mul!, adjoint, Adjoint, Transpose, empty_array
using MatrixCombinators.LinearAlgebra: transpose,
    A_mul_B!, A_mul_Bt!, At_mul_Bt!, A_mul_Bc!, Ac_mul_Bc!

eager_t = Dict('N' => identity, 'C' => adjoint, 'T' => transpose)
lazy_t = Dict('N' => identity, 'C' => Adjoint, 'T' => Transpose)

"""
Combination of pairs of N, C, T excluding (C, T) and (T, C).
"""
t_pairs = [(cA, cB) for cA in "NCT", cB in "NCT"
           if cA == 'N' || cB == 'N' || cA == cB]
