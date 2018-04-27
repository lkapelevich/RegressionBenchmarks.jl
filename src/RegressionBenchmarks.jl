module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO, MLDataUtils
import Base.rand

export BinChoice, NoNoise, MatrixCorrelation, NoCorrelation,
    getw, getX, getdata, BenchmarkData, XData,
    RegressionMethod, ExactPrimalCuttingPlane,
    validate_params!,
    solve_problem,
    isRsquared, oosRsquared, accuracy, falsepositive, predict_sparse,
    normalize!

include("datatypes.jl")
include("data.jl")
include("utils.jl")
include("methods.jl")
include("validation.jl")

# TODO try cutting plane with user heuristic. cache subgradient in extension dictionary
# TODO lagrangian relaxation relaxing knapsack constraint
# TODO different solvers should enter different methods

end # module
