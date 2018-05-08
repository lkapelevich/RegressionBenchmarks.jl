module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO,
      MLDataUtils, JuMP, Gurobi, CPLEX, MathProgBase
using Gadfly, DataFrames
import Base.rand #, Base.normalize!
import Base.mkdir

export BinChoice, NoNoise, MatrixCorrelation, NoCorrelation,
    getw, getX, getdata, BenchmarkData, XData,
    # Methods
    RegressionMethod, ExactPrimalCuttingPlane, PrimalWithHeuristics,
    RelaxDualSubgradient, RelaxDualCutting,
    # Parts of methods
    ConstantStepping, PolyakStepping,
    # Validate and solve
    validate_params!,
    solve_problem,
    # Utilities
    isRsquared, oosRsquared, accuracy, falsepositive, predict_sparse,
    # Results
    benchmark, makeplot,
    data2str, method2str #,
    # normalize!

const YVector = Union{Vector{Float64},SubArray{Float64,1}}

include("datatypes.jl")
include("data.jl")
include("methods/methods.jl")
include("utils.jl")
include("validation.jl")
include("benchmark.jl")
include("plots.jl")

# TODO keyword argument in sort function type unstable make it sort -ax

# TODO try cutting plane with user heuristic. cache subgradient in extension dictionary
# TODO lagrangian relaxation relaxing knapsack constraint
# TODO different solvers should enter different methods
# TODO subgradient with varying stepsizes
# TODO validation results to file

end # module
