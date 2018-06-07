added(A, B) = AddedMatrices(A, B)
added(A::UniformScaling, B::UniformScaling) = A + B
added(a::Number, B) = added(a * I, B)
added(A, b::Number) = added(A, b * I)
added(a::Number, b::Number) = (a + b) * I  # disambiguate

muled(A, B) = MultipliedMatrices(A, B)
