LinearAlgebra.mul!(Y,
                   A::Union{PairedMatrices,
                            Adjoint{<: PairedMatrices},
                            Transpose{<: PairedMatrices},
                            # SubArray{, , <: PairedMatrices},
                            },
                   B) = _mul!(Y, A, B)
