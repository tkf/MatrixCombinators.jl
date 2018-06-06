include("preamble.jl")
using BenchmarkTools
using MatrixCombinators: muled, empty_array, mul!, _mul!, DumbExecutor

suite = BenchmarkGroup()
suite["default"] = suite_default = BenchmarkGroup()
suite["dumb"] = suite_dumb = BenchmarkGroup()
# suite_manual = BenchmarkGroup()


function benchmarkables_DiagTimesCSC(n, p, xtype, q=0.1)
    if xtype <: Vector
        make_x = (rng, n) -> randn(rng, n)
    elseif xtype <: SparseVector
        make_x = (rng, n) -> sprand(rng, n, q)
    else
        error("Unknown x type: $xtype")
    end

    prepare = function(rng)
        D = Diagonal(randn(rng, n))
        S = sprand(rng, n, n, p)
        x = make_x(rng, n)
        y = empty_array(Vector{Float64}, n)
        M = muled(D, S)
        return y, M, x
    end

    bench_default = @benchmarkable(
        mul!(y, M, x),
        setup = begin
        y, M, x = $prepare($(MersenneTwister(1)))
        end)

    bench_dumb = @benchmarkable(
        _mul!(executor, y, M, x),
        setup = begin
        y, M, x = $prepare($(MersenneTwister(1)))
        executor = DumbExecutor(M.executor.allocator)
        end)

    return bench_default, bench_dumb
end


let bg_default = suite_default["DiagTimesCSC"] = BenchmarkGroup()
    bg_dumb = suite_dumb["DiagTimesCSC"] = BenchmarkGroup()
    # bg_manual = suite_manual["DiagTimesCSC"] = BenchmarkGroup()

    for args in [
            (1000, 0.1, Vector),
            (1000, 0.01, Vector),
            (1000, 0.001, Vector),
            (1000, 0.1, SparseVector),
            ]
        # bg_default[args], bg_dumb[args], bg_manual[args] =
        bg_default[args], bg_dumb[args] =
            benchmarkables_DiagTimesCSC(args...)
    end
end


suite
