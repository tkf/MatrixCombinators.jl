@static if VERSION >= v"0.7.0-"
    const BLAS = LinearAlgebra.BLAS
end

"""
    _gemm!(tA, tB, α, A, B, β, C)

`BLAS.gemm!` wrapper; i.e., it does `C = α * Â * B̂ + β * C` where `Â`
(`B̂`) is `A` (`B`) itself or its transpose or adjoint depending on the
value of `tA` (`tB`).

See also:

- [LinearAlgebra.BLAS.gemm!](https://docs.julialang.org/en/latest/stdlib/LinearAlgebra/#LinearAlgebra.BLAS.gemm!)

`BLAS.gemm!` is called if `A` is one of:

- [BlockBandedMatrices.BlockBandedBlock](https://github.com/JuliaMatrices/BlockBandedMatrices.jl/blob/v0.0.1/src/linalg.jl#L60)
- [Array](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/LinearAlgebra/src/blas.jl#L1084)
"""
function _gemm!(tA, tB,
                alpha::T,
                A::AbstractVecOrMat{T},
                B::AbstractVecOrMat{T},
                beta::T,
                C::AbstractVecOrMat{T},
                ) where {T <: LinearAlgebra.BlasFloat}

    # Taken from LinearAlgebra.gemm_wrapper!:
    mA, nA = LinearAlgebra.lapack_size(tA, A)
    mB, nB = LinearAlgebra.lapack_size(tB, B)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
end


"""
    _gemv!(tA, α, A, X, β, Y)

`BLAS.gemv!` wrapper; i.e., it does `y = α * Â * x + β * y` where `Â`
is `A` itself or its transpose or adjoint depending on the value of
`tA`.

See also:

- [LinearAlgebra.BLAS.gemv!](https://docs.julialang.org/en/latest/stdlib/LinearAlgebra/#LinearAlgebra.BLAS.gemv!)
"""
_gemv!(tA, alpha, A, X, beta, Y) = BLAS.gemv!(tA, alpha, A, X, beta, Y)


"""
When `A` isa `SparseMatrixCSC`, `_gemm!` calls `SparseMatrixCSC`'s
GEMM-like interface.

See:
- [base/sparse/linalg.jl (v0.6.3)](https://github.com/JuliaLang/julia/blob/v0.6.3/base/sparse/linalg.jl#L46)
- [stdlib/SparseArrays/src/linalg.jl (v0.7.0-alpha)](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/SparseArrays/src/linalg.jl#L32)
"""
function _gemm!(tA, tB,
                alpha::Number,
                A::SparseMatrixCSC,
                B::StridedVecOrMat,
                beta::Number,
                C::StridedVecOrMat)
    @assert tA in ('N', 'C', 'T')
    @assert tB == 'N'
    if tA == 'N'
        A_mul_B!(alpha, A, B, beta, C)
    elseif tA == 'C'
        Ac_mul_B!(alpha, A, B, beta, C)
    elseif tA == 'T'
        At_mul_B!(alpha, A, B, beta, C)
    end
end

"""
When `A` isa `SparseMatrixCSC`, `_gemv!` calls `SparseMatrixCSC`'s
GEMM-like interface (via `_gemm!`).

See:
- [base/sparse/sparsevector.jl (v0.6.3)](https://github.com/JuliaLang/julia/blob/v0.6.3/base/sparse/sparsevector.jl#L1532)
- [stdlib/SparseArrays/src/sparsevector.jl (v0.7.0-alpha)](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/SparseArrays/src/sparsevector.jl#L1477)
"""
function _gemv!(tA,
                alpha::Number,
                A::SparseMatrixCSC,
                X::StridedVector,
                beta::Number,
                Y::StridedVector)
    _gemm!(tA, 'N', alpha, A, X, beta, Y)
end

"""
    gmul!(C, A, B, α, β)

GEMM/GEMV-like multiplication ("generalized" multiplication) interface.
It calculates:

    C = α * A * B + β * C

Its signature is based on `SparseMatrixCSC`'s `mul!`:
- [stdlib/SparseArrays/src/linalg.jl (v0.7.0-alpha)](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/SparseArrays/src/linalg.jl#L32)
"""
function gmul!(Y::AbstractMatrix{TY},
               A::AbstractMatrix{TA},
               B::AbstractMatrix{TB},
               alpha = one(promote_type(TY, TA, TB)),
               beta = one(promote_type(TY, TA, TB)),
               ) where {TY, TA, TB}
    tA = tchar(A)
    tB = tchar(B)
    _gemm!(tA, tB, alpha, peel(A), peel(B), beta, Y)
end

function gmul!(Y::AbstractVector{TY},
               A::AbstractMatrix{TA},
               B::AbstractVector{TB},
               alpha = one(promote_type(TY, TA, TB)),
               beta = one(promote_type(TY, TA, TB)),
               ) where {TY, TA, TB}
    tA = tchar(A)
    _gemv!(tA, alpha, peel(A), B, beta, Y)
end

# Short-circuit _gemm! for sparse matrix/vector in Julia 0.7
@static if VERSION >= v"0.7.0-"
"""
When `A` isa `SparseMatrixCSC`, `gmul!` directly calls
`SparseMatrixCSC`'s GEMM-like interface.

See:
- [stdlib/SparseArrays/src/linalg.jl (v0.7.0-alpha)](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/SparseArrays/src/linalg.jl#L32)
"""
function gmul!(Y::StridedVecOrMat,
               A::Union{SparseMatrixCSC,
                        Adjoint{<: Any, <: SparseMatrixCSC},
                        Transpose{<: Any, <: SparseMatrixCSC}},
               B::StridedVecOrMat,
               alpha,
               beta,
               )
    mul!(Y, A, B, alpha, beta)
end

"""
When `B` isa `AbstractSparseVector`, `gmul!` directly calls
`SparseMatrixCSC`'s GEMM-like interface.

See:
- [stdlib/SparseArrays/src/sparsevector.jl (v0.7.0-alpha)](https://github.com/JuliaLang/julia/blob/v0.7.0-alpha/stdlib/SparseArrays/src/sparsevector.jl#L1477)
"""
function gmul!(Y::AbstractVector,
               A::StridedMatrix,
               B::AbstractSparseVector,
               alpha,
               beta,
               )
    mul!(Y, A, B, alpha, beta)
end
end


"""
    amul!(Y, A, B)

A short-hand notation for "`gmul!(Y, A, B, 1, 1)`", i.e., it does:

    Y += A * B
"""
function amul!(Y::AbstractArray{TY},
               A::AbstractMatrix{TA},
               B::AbstractArray{TB}) where {TY, TA, TB}
    alpha = beta = one(promote_type(TY, TA, TB))
    return gmul!(Y, A, B, alpha, beta)
end


has_gemm(::Type, ::Type, ::Type) = false
has_gemv(::Type, ::Type, ::Type) = false


function has_gemm(TA::Type{<: Union{AbstractVecOrMat{T},
                                    Adjoint{T, <: AbstractVecOrMat{T}},
                                    Transpose{T, <: AbstractVecOrMat{T}}}},
                  TB::Type{<: Union{AbstractVecOrMat{T},
                                    Adjoint{T, <: AbstractVecOrMat{T}},
                                    Transpose{T, <: AbstractVecOrMat{T}}}},
                  TC::Type{<: AbstractVecOrMat{T}},
                  ) where {T <: BLAS.BlasFloat}
    et = promote_type(eltype(TA), eltype(TB), eltype(TC))
    return length(methods(BLAS.gemm!, (Char, Char, et, TA, TB, et, TC))) > 0
end


function has_gemv(TA::Type{<: Union{AbstractVecOrMat{T},
                                    Adjoint{T, <: AbstractVecOrMat{T}},
                                    Transpose{T, <: AbstractVecOrMat{T}}}},
                  TB::Type{<: AbstractVecOrMat{T}},
                  TC::Type{<: AbstractVector{T}},
                  ) where {T <: BLAS.BlasFloat}
    et = promote_type(eltype(TA), eltype(TB), eltype(TC))
    return length(methods(BLAS.gemv!, (Char, et, TA, TB, et, TC))) > 0
end


has_gemm(TA::Type{<: SparseMatrixCSC},
         TB::Type{<: StridedVecOrMat},
         TC::Type{<: StridedVecOrMat}) = true

has_gemv(TA::Type{<: SparseMatrixCSC},
         TB::Type{<: StridedVector},
         TC::Type{<: StridedVector}) = true


has_gmul(TC::Type{<: AbstractMatrix}, TA::Type, TB::Type) = has_gemm(TA, TB, TC)
has_gmul(TC::Type{<: AbstractVector}, TA::Type, TB::Type) = has_gemv(TA, TB, TC)
has_gmul(C, A, B) = has_gmul(typeof.((C, A, B))...)

has_mul(::Type, ::Type, ::Type) = false
has_mul(::Type{<: AbstractVector},
        ::Type{<: AbstractMatrix},
        ::Type{<: AbstractVector}) = true
has_mul(::Type{<: AbstractMatrix},
        ::Type{<: AbstractMatrix},
        ::Type{<: AbstractMatrix}) = true
has_mul(C, A, B) = has_mul(typeof.((C, A, B))...)
