module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO, MLDataUtils
import Base.rand

export BinChoice, NoNoise,
    getw, getX, getdata, BenchmarkData,
    RegressionMethod, ExactPrimalCuttingPlane,
    validate_params!,
    solve_problem,
    isRsquared, oosRsquared, accuracy, falsepositive, predict_sparse

include("datatypes.jl")
include("utils.jl")
include("data.jl")
include("methods.jl")
include("validation.jl")

end # module
