_ops = :(
    LinearAlgebra.A_mul_B!,
    LinearAlgebra.A_mul_Bt!, LinearAlgebra.At_mul_B!, LinearAlgebra.At_mul_Bt!,
    LinearAlgebra.A_mul_Bc!, LinearAlgebra.Ac_mul_B!, LinearAlgebra.Ac_mul_Bc!,
).args
for f in _ops
    name = string(f.args[end].value)  # e.g., "A_mul_B!"
    @assert parse("LinearAlgebra.$name") == f

    if startswith(name, "A_")
        TrM = identity
    elseif startswith(name, "Ac_")
        TrM = Adjoint
    elseif startswith(name, "At_")
        TrM = Transpose
    else
        error("Unknown: $name")
    end

    if endswith(name, "_B!")
        TrX = identity
    elseif endswith(name, "_Bc!")
        TrX = Adjoint
    elseif endswith(name, "_Bt!")
        TrX = Transpose
    else
        error("Unknown: $name")
    end

    @eval $f(Y::AbstractMatrix, M::PairedMatrices, X::AbstractMatrix) =
        _mul!(Y, $TrM(M), $TrX(X))
    if TrX == identity
        @eval $f(Y::AbstractVector, M::PairedMatrices, X::AbstractVector) =
            _mul!(Y, $TrM(M), X)
    end
end
