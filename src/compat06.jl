struct Adjoint{T,S} <: AbstractMatrix{T}
    parent::S
    function Adjoint{T,S}(A::S) where {T,S}
        new(A)
    end
end

struct Transpose{T,S} <: AbstractMatrix{T}
    parent::S
    function Transpose{T,S}(A::S) where {T,S}
        new(A)
    end
end

Adjoint(A::S) where {T, S <: AbstractMatrix{T}} = Adjoint{T,S}(A)
Transpose(A::S) where {T, S <: AbstractMatrix{T}} = Transpose{T,S}(A)

const AdjOrTrans = Union{Adjoint, Transpose}

Base.size(A::AdjOrTrans) = size(A.parent)
Base.getindex(A::AdjOrTrans, i::Int) = A.parent[i]
Base.getindex(A::AdjOrTrans, I::Vararg{Int, N}) where N =
    getindex(A.parent, I...)

_tstr(::Type{<: Adjoint}) = "c"
_tstr(::Type{<: Transpose}) = "t"
_tstr(::Type) = ""

@generated function mul!(Y, A, B)
    tA = _tstr(A)
    tB = _tstr(B)
    f = getfield(LinearAlgebra, Symbol("A$(tA)_mul_B$(tB)!"))
    a = A <: Union{Adjoint, Transpose} ? :(A.parent) : :A
    b = B <: Union{Adjoint, Transpose} ? :(B.parent) : :B
    :($f(Y, $a, $b))
end
