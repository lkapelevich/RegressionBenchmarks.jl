module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO, MLDataUtils
using Gadfly, DataFrames
import Base.rand #, Base.normalize!
import Base.mkdir

export BinChoice, NoNoise, MatrixCorrelation, NoCorrelation,
    getw, getX, getdata, BenchmarkData, XData,
    RegressionMethod, ExactPrimalCuttingPlane,
    validate_params!,
    solve_problem,
    isRsquared, oosRsquared, accuracy, falsepositive, predict_sparse,
    benchmark, makeplot #,
    # normalize!

include("datatypes.jl")
include("data.jl")
include("utils.jl")
include("methods.jl")
include("validation.jl")
include("benchmark.jl")
include("plots.jl")

# TODO try cutting plane with user heuristic. cache subgradient in extension dictionary
# TODO lagrangian relaxation relaxing knapsack constraint
# TODO different solvers should enter different methods
# TODO subgradient with varying stepsizes

end # module
