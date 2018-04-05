module RegressionBenchmarks

using StatsBase, Distributions
import Base.rand

export BinChoice,
    getw, getX

struct BinChoice <: DiscreteUnivariateDistribution end

rand(::BinChoice) = sample([-1, 1])

const BenchmarkXDists = Union{Normal, MvNormal, Uniform}
const BenchmarkwDists = Union{BinChoice, Normal, Uniform}

struct BenchmarkData
    Xdist::BenchmarkXDists
    wdist::BenchmarkwDists
    p::Int
end

function getw(d::Int, k::Int, wdist::BenchmarkwDists)
    w = zeros(d)
    support = sample(1:d, k, replace=false)
    w[support] = rand(wdist, k)
    w
end

function getX(n::Int, d::Int, Xdist::BenchmarkXDists)
    rand(Xdist, n, d)
end
function getX(n::Int, d::Int, Xdist::MvNormal)
    rand(Xdist, n)'
end

end # module
