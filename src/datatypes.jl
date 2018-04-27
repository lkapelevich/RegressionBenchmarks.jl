import Distributions.MvNormal

struct BinChoice <: DiscreteUnivariateDistribution end
struct NoNoise <: DiscreteUnivariateDistribution end

rand(::BinChoice) = sample([-1, 1])
rand(::NoNoise) = 0.0
rand(::NoNoise, i::Int) = zeros(i)

const BenchmarkXDists = Union{Normal, MvNormal, Uniform}
const BenchmarkwDists = Union{BinChoice, Normal, Uniform}
const BenchmarkNoiseDists = Union{NoNoise, Normal}

abstract type XCorrelation end
struct NoCorrelation <: XCorrelation end
struct MatrixCorrelation <: XCorrelation
    coeff::Float64
end

struct XData{T <: BenchmarkXDists}
    dist::T
    corr::XCorrelation
end
XData() = XData(MvNormal, NoCorrelation())

struct BenchmarkData
    Xdata::XData
    wdist::BenchmarkwDists
    noisedist::BenchmarkNoiseDists
    SNR::Float64
    n::Int
    d::Int
    k::Int
end

"""
    XData(::Type{T}, c::XCorrelation, d::Int) where {T <: BenchmarkXDists}

Instantiates a distribution of type `T` and modifies it,so that we follow
 correlation type `c`. Returns type `XData`, caching the distribution and
correlation type we used.
"""
function XData(::Type{T}, c::XCorrelation, d::Int) where {T <: BenchmarkXDists}
    error("Need to implement method XData for $T and $c.")
end
function XData(::Type{MvNormal}, c::NoCorrelation, d::Int)
    XData(MvNormal(zeros(d), eye(d)), c)
end
function XData(::Type{MvNormal}, c::MatrixCorrelation, d::Int)
    Σ = c.coeff * ones(d, d)
    @inbounds for i = 1:d
        Σ[i, i] = 1.0
    end
    XData(MvNormal(zeros(d), Σ), c)
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

function getX(n::Int, d::Int, Xdata::XData)
    rand(Xdata.dist, n, d)
end
function getX(n::Int, ::Int, Xdata::XData{T}) where {T <: MvNormal}
    rand(Xdata.dist, n)'
end
function getX(bd::BenchmarkData)
    getX(bd.n, bd.d, bd.Xdata)
end

function getnoise(bnd::NoNoise, ::Float64, Y::Vector{Float64})
    n = length(Y)
    zeros(n)
end
function getnoise(bnd::BenchmarkNoiseDists, SNR::Float64, Y::Vector{Float64})
    n = length(Y)
    noise = rand(bnd, n)
    norm(noise) < 1e-3 && return noise
    SNR < 1e-3 && error("Your SNR is too small.")
    noise .*= (norm(Y) / ( SNR * norm(noise) ) )
end
function getnoise(bd::BenchmarkData, Y::Vector{Float64})
    getnoise(bd.noisedist, bd.SNR, Y)
end
