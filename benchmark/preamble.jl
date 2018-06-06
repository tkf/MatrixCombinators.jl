@static if VERSION < v"0.7.0-"
    const ComplexF64 = Complex128
    const ComplexF32 = Complex64
else
    using Random
end
