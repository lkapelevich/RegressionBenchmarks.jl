struct BinChoice <: DiscreteUnivariateDistribution end
struct NoNoise <: DiscreteUnivariateDistribution end

rand(::BinChoice) = sample([-1, 1])
rand(::NoNoise) = 0.0
rand(::NoNoise, i::Int) = zeros(i)

const BenchmarkXDists = Union{Normal, MvNormal, Uniform}
const BenchmarkwDists = Union{BinChoice, Normal, Uniform}
const BenchmarkNoiseDists = Union{NoNoise, Normal}

struct BenchmarkData
    Xdist::BenchmarkXDists
    wdist::BenchmarkwDists
    noisedist::BenchmarkNoiseDists
    SNR::Float64
    n::Int
    d::Int
    k::Int
end

function getw(d::Int, k::Int, wdist::BenchmarkwDists)
    w = zeros(d)
    support = sample(1:d, k, replace=false)
    w[support] = rand(wdist, k)
    w
end
function getw(bd::BenchmarkData)
    getw(bd.d, bd.k, bd.wdist)
end

function getX(n::Int, d::Int, Xdist::BenchmarkXDists)
    rand(Xdist, n, d)
end
function getX(n::Int, d::Int, Xdist::MvNormal)
    rand(Xdist, n)'
end
function getX(bd::BenchmarkData)
    getX(bd.n, bd.d, bd.Xdist)
end

function getnoise(n::Int, nd::BenchmarkNoiseDists)
    rand(nd, n)
end
function getnoise(bd::BenchmarkData)
    getnoise(bd.n, bd.noisedist)
end
