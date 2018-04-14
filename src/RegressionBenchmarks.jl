module RegressionBenchmarks

using StatsBase, Distributions, SubsetSelection, SubsetSelectionCIO
import Base.rand

export BinChoice,
    getw, getX, getdata

include("datatypes.jl")
include("data.jl")

end # module
