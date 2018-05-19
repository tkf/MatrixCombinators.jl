@static if VERSION < v"0.7.0-"
    const LinearAlgebra = Base.LinAlg
else
    using LinearAlgebra
end
const A_mul_B! = LinearAlgebra.A_mul_B!
