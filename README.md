# MatrixCombinators --- lazily add or multiply matrices

[![Build Status][travis-img]][travis-url]
[![Coverage Status][coveralls-img]][coveralls-url]
[![codecov.io][codecov-img]][codecov-url]

`MatrixCombinators` package provide functions
`MatrixCombinators.added` and `MatrixCombinators.muled` to create a
"lazy" matrix representing addition and multiplication of two
matrices, respectively.  It is useful for computations with a
structured matrix which can be represented as additions and
multiplications of sparse and/or structured matrices.  For example,
[small-world network](#small-world-network-example) can be represented
as a sum of banded and sparse matrices.


## Usage

`MatrixCombinators.added` creates a "lazy" matrix representing
addition of two matrices:

```julia
using MatrixCombinators

A = [1 2
     3 4]
B = [4 5
     6 7]
M = MatrixCombinators.added(A, B)
```

Here, `M` is a lazy matrix which behaves as a standard non-lazy matrix

```julia
D = A .+ B
```

but without actually calculating `A + B` at the time it is created.
Rather, it "happens" at the time it is used in computation:

```julia
x = ones(2)
@test M * x == D * x
```

Here, `(A * x) + (B * x)` is actually computed instead.
Multiplication can be more efficient when underlying matrices `A` and
`B` have specific structures (e.g., `SparseMatrixCSC`,
`BandedMatrix`).  In this case, using `A*_mul_B*!` variants makes more
sense to avoid memory-allocation.  Indeed, `MatrixCombinators`
supports all variants (but see [Limitations](#limitations)):

```julia
y = similar(x)
A_mul_B!(y, M, x)
@test y == D * x
```

`MatrixCombinators.muled` works similarly for matrix multiplication:

```julia
M = MatrixCombinators.muled(A, B)
D = A * B
A_mul_B!(y, M, x)
@test y == D * x
```

That is to say, `M * x` computes `A * (B * x)`.


### Internal temporary array (`b_out` cache)

Lazy matrices created by `added` and `muled` can only be multiplied by
a vector (at the moment; see [Limitations](#limitations)).  This is
because those lazy matrices carry a cache array (`b_out`) for storing
temporary result of `B * x` calculation.

To use matrix-matrix multiplication lazy matrices created by `added`
and `muled`, pass temporary arrays to them as the third argument:

```julia
b_out = typeof(B)((2, 3))
M = MatrixCombinators.added(A, B, b_out)

X = [0 1 1
     1 0 1]
Y = similar(X)
A_mul_B!(Y, M, X)
@test Y == (A .+ B) * X
```

### Small-world network example

<!--
```julia
@static if Pkg.installed("BandedMatrices") != nothing
```
-->

Here is an example to create a directed small-world network with
random weights:

```julia
using BandedMatrices
using IterTools: product

n = 1000
A = BandedMatrix(Zeros(n, n), (1, 1))
B = spzeros(n, n)

# Fill off-diagonal parts:
A[CartesianIndex.(2:n, 1:n-1)] = randn(n - 1)
A[CartesianIndex.(1:n-1, 2:n)] = randn(n - 1)

# Make a loop
B[end, 1] = randn()
B[1, end] = randn()

# Randomly connect nodes
all_indices = collect(product(1:n, 1:n))
m = ceil(Int, length(all_indices) * 0.01)
idx = rand(all_indices, m)
B[CartesianIndex.(idx)] = randn(length(idx))

M = MatrixCombinators.added(A, B)
```

<!--
```julia
end
```
-->


## Limitations

(Those are not "theoretical" limitation and are solvable, well, with
more code...)

* `At_mul_B*!` and `Ac_mul_B*!` work only when first matrix passed to
  `added` and `muled` is a square matrix.

* `b_out` cache must be specified at creation time in order to use
  `A*_mul_B*!` with matrix as the second argument.

* No support for `A_ldiv_B!` etc.


[travis-img]: https://travis-ci.org/tkf/MatrixCombinators.jl.svg?branch=master
[travis-url]: https://travis-ci.org/tkf/MatrixCombinators.jl
[coveralls-img]: https://coveralls.io/repos/tkf/MatrixCombinators.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/tkf/MatrixCombinators.jl?branch=master
[codecov-img]: http://codecov.io/github/tkf/MatrixCombinators.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/tkf/MatrixCombinators.jl?branch=master
