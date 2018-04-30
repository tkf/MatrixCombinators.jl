@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using MatrixCombinators
using MatrixCombinators.LinearAlgebra:
    A_mul_B!, A_mul_Bt!, At_mul_Bt!, A_mul_Bc!, Ac_mul_Bc!
