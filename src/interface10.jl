LinearAlgebra.mul!(Y::AbstractMatrix, A::AdjTrOrPair, B::AbstractMatrix) =
    _mul!(Y, A, B)

LinearAlgebra.mul!(Y::AbstractVector, A::AdjTrOrPair, B::AbstractVector) =
    _mul!(Y, A, B)

# Disambiguation:
LinearAlgebra.mul!(Y::AbstractMatrix,
                   A::AdjTrOrPair,
                   B::Adjoint{<: Any, <: AbstractMatrix}) =
    _mul!(Y, A, B)

LinearAlgebra.mul!(Y::AbstractMatrix,
                   A::AdjTrOrPair,
                   B::Transpose{<: Any, <: AbstractMatrix}) =
    _mul!(Y, A, B)
