struct BinChoice <: DiscreteUnivariateDistribution end

rand(::BinChoice) = sample([-1, 1])

const BenchmarkXDists = Union{Normal, MvNormal, Uniform}
const BenchmarkwDists = Union{BinChoice, Normal, Uniform}

struct BenchmarkData
    Xdist::BenchmarkXDists
    wdist::BenchmarkwDists
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
