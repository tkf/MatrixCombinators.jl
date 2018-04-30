module MatrixCombinators

abstract type PairedMatrices{TA, TB, BO} end

struct AddedMatrices{TA, TB, BO} <: PairedMatrices{TA, TB, BO}
    A::TA
    B::TB
    b_out::BO

    function AddedMatrices(A::TA, B::TB,
                           b_out::BO = zeros(eltype(B), size(B, 1)),
                           ) where {TA, TB, BO}
        @assert size(A) == size(B)
        return new{TA, TB, BO}(A, B, b_out)
    end
end

const added = AddedMatrices

struct MultipliedMatrices{TA, TB, BO} <: PairedMatrices{TA, TB, BO}
    A::TA
    B::TB
    b_out::BO

    function MultipliedMatrices(A::TA, B::TB,
                                b_out::BO = zeros(eltype(B), size(B, 1)),
                                ) where {TA, TB, BO}
        @assert size(A, 2) == size(B, 1)
        return new{TA, TB, BO}(A, B, b_out)
    end
end

const muled = MultipliedMatrices


# --- Utilities

const PMMatOut{TA, TB} = PairedMatrices{TA, TB, <: AbstractMatrix}
const PMVecOut{TA, TB} = PairedMatrices{TA, TB, <: AbstractVector}

const AddMatOut{TA, TB} = AddedMatrices{TA, TB, <: AbstractMatrix}
# const AddVecOut{TA, TB} = AddedMatrices{TA, TB, <: AbstractVector}

const MulMatOut{TA, TB} = MultipliedMatrices{TA, TB, <: AbstractMatrix}
# const MulVecOut{TA, TB} = MultipliedMatrices{TA, TB, <: AbstractVector}

b_out_vec(M::PMMatOut) = @view M.b_out[:, 1]
b_out_vec(M::PMVecOut) = M.b_out

empty_array(::Type{T}, dims) where {T <: AbstractArray} = T(dims)
empty_array(::Type{<: SparseMatrixCSC{T}}, dims,) where {T} =
    spzeros(T, dims...)


# --- Array Interface
# https://docs.julialang.org/en/stable/manual/interfaces/#man-interface-array-1

Base.size(M::AddedMatrices, dim...) = size(M.A, dim...)
Base.getindex(M::AddedMatrices, i::Int) = M.A[i] + M.B[i]
Base.getindex(M::AddedMatrices, I::Vararg{Int, N}) where N =
    getindex(M.A, I...) +
    getindex(M.B, I...)

preferred_style(t::T, ::T) where {T <: IndexStyle} = t
preferred_style(::IndexStyle, ::IndexStyle) = IndexCartesian()

Base.IndexStyle(::Type{<:AddedMatrices{TA, TB}}) where {TA, TB} =
    preferred_style(IndexStyle(TA), IndexStyle(TB))

function Base.convert(T::Type{<: AbstractArray}, M::AddedMatrices)
    Y = empty_array(T, size(M))
    @. Y = M.A + M.B
    return Y
end

Base.size(M::MultipliedMatrices) = (size(M.A, 1), size(M.B, 2))

function Base.size(M::MultipliedMatrices, i::Integer)
    if i == 1
        size(M.A, 1)
    else
        size(M.B, i)  # if i > 2, this returns 1
    end
end

function Base.getindex(M::MultipliedMatrices, i::Int, j::Int)
    x = spzeros(Int, size(M, 2))
    x[j] = 1
    return (M * x)[j]
end

Base.IndexStyle(::Type{<:MultipliedMatrices}) = IndexCartesian()

function Base.convert(T::Type{<: AbstractArray}, M::MultipliedMatrices)
    Y = empty_array(T, size(M))
    A_mul_B!(Y, M.A, M.B)
    return Y
end


# --- Computations

# AddedMatrices
added_ops = :(
    Base.A_mul_B!,
    Base.A_mul_Bt!, Base.At_mul_B!, Base.At_mul_Bt!,
    Base.A_mul_Bc!, Base.Ac_mul_B!, Base.Ac_mul_Bc!,
).args
for (TY, TM, b_out) in
        [(AbstractMatrix, AddMatOut, :(M.b_out)),
         (AbstractVector, AddedMatrices, :(b_out_vec(M)))]
    for f in added_ops
        @eval function $f(Y::$TY, M::$TM, X)
            b_out = $b_out
            $f(Y, M.A, X)
            $f(b_out, M.B, X)
            Y .+= b_out
            return Y
        end
    end
end

# MultipliedMatrices
muled_ops = :(
    Base.A_mul_B!,
    Base.A_mul_Bt!,
    Base.A_mul_Bc!,
).args
for (TY, TM, b_out) in
        [(AbstractMatrix, MulMatOut, :(M.b_out)),
         (AbstractVector, MultipliedMatrices, :(b_out_vec(M)))]
    for f in muled_ops
        @eval function $f(Y::$TY, M::$TM, X)
            b_out = $b_out
            $f(b_out, M.B, X)
            $f(Y, M.A, b_out)
            return Y
        end
    end
end

function Base.:*(M::PairedMatrices, x::AbstractVector)
    y = similar(@view M.A[:, 1])
    A_mul_B!(y, M, x)
    return y
end

function Base.:*(M::PairedMatrices, X::AbstractMatrix)
    Y = similar(M.A, (size(M, 1), size(X, 2)))  # TODO: improve
    A_mul_B!(Y, M, X)
    return Y
end

end # module
