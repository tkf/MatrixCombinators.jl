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
    @test_deprecated07 @test size(added(I, ones(2, 3)), 2, 1) == (3, 2)

    @test added(ones(Int, 2, 3), 4) isa AddedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test added(ones(Int, 2, 3), I) isa AddedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test size(added(ones(2, 3), I)) == (2, 3)
    @test_deprecated07 @test size(added(ones(2, 3), I), 2, 1) == (3, 2)
end

@testset "muled" begin
    muled = MatrixCombinators.muled
    MultipliedMatrices = MatrixCombinators.MultipliedMatrices

    @test muled(2I, 3I)::UniformScaling == 6I
    @test muled(2I, 3)::UniformScaling == 6I
    @test muled(2, 3I)::UniformScaling == 6I
    @test muled(2, 3)::UniformScaling == 6I

    @test muled(1, ones(2, 3)) isa MultipliedMatrices{Float64, <:UniformScaling}
    @test muled(I, ones(2, 3)) isa MultipliedMatrices{Float64, <:UniformScaling}
    @test size(muled(I, ones(2, 3))) == (2, 3)
    @test_deprecated07 @test size(muled(I, ones(2, 3)), 2, 1) == (3, 2)

    @test muled(ones(Int, 2, 3), 4) isa MultipliedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test muled(ones(Int, 2, 3), I) isa MultipliedMatrices{Int, <:Matrix,
                                                      <:UniformScaling}
    @test size(muled(ones(2, 3), I)) == (2, 3)
    @test_deprecated07 @test size(muled(ones(2, 3), I), 2, 1) == (3, 2)
end
