# X Uniform

# X Normal


# Vary autocorrelation

# Vary noise

# W uniform


# W normal


# W multichoice

# logistic regression rather than regression

type RegressionData
    X::Array{Float64,2}
    Y::Vector{Float64}
    w::Vector{Float64}
end

"""
    function getdata(; Xdist::DataType{XDist} = Normal,
                         wdist::DataType{wdist} = BinChoice,
                         n::Int = 100,
                         p::Int = 100,
                         k::Int = 10
                        ) where {XDist <: BenchmarkXDists, wdist <: BenchmarkwDists}
"""
function getdata(; Xdist::BenchmarkXDists = Normal(),
                         wdist::BenchmarkwDists = BinChoice(),
                         n::Int = 100,
                         p::Int = 100,
                         k::Int = 10
                        )
    bd = BenchmarkData(Xdist, wdist, n, p, k)
    X = getX(bd)
    w = getw(bd)
    Y = X * w
    RegressionData(X, Y, w)
end
function getdata(; Xdist::BenchmarkXDists = Normal(),
                         wdist::BenchmarkwDists = BinChoice(),
                         n::Int = 100,
                         p::Int = 100,
                         k::Int = 10
                        )
    bd = BenchmarkData(Xdist, wdist, n, p, k)
    w = getw(bd)
    X = getX(bd)
    size(X) == (n, p) || error("X distribution doesn't match dimensions requested.")
    Y = X * w
    RegressionData(X, Y, w)
end
