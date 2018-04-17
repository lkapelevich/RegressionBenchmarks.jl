module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO, MLDataUtils
import Base.rand

export BinChoice, NoNoise,
    getw, getX, getdata, BenchmarkData,
    RegressionMethod, ExactPrimalCuttingPlane,
    validate_params!,
    solve_problem

include("datatypes.jl")
include("data.jl")
include("methods.jl")
include("validation.jl")

end # module
