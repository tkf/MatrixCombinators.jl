include("preamble.jl")

@testset "added" begin
    added = MatrixCombinators.added
    AddedMatrices = MatrixCombinators.AddedMatrices

    @test added(2I, 3I)::UniformScaling == 5I
    @test added(2I, 3)::UniformScaling == 5I
    @test added(2, 3I)::UniformScaling == 5I
    @test added(2, 3)::UniformScaling == 5I

    @test added(1, ones(2, 3)) isa AddedMatrices{Float64, <:UniformScaling}
    @test added(I, ones(2, 3)) isa AddedMatrices{Float64, <:UniformScaling}
    @test size(added(I, ones(2, 3))) == (2, 3)
    @test size(added(I, ones(2, 3)), 2, 1) == (3, 2)

    @test added(ones(Int, 2, 3), 4) isa AddedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test added(ones(Int, 2, 3), I) isa AddedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test size(added(ones(2, 3), I)) == (2, 3)
    @test size(added(ones(2, 3), I), 2, 1) == (3, 2)
end
