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

"""
    BenchmarkData(; Xdata::XData = XData(),
                         wdist::BenchmarkwDists = BinChoice(),
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 20.0,
                         n::Vector{Int} = 100,
                         nfeatures::Int = 100,
                         k::Int = 10
                    )

Constructs a data object that fully describes the regression task as far as
this library is concerned.

# Arguments
* Xdata: how synthetic independent variables will be constructed (an object
of type `XData` has a distribution and a correlation mechanism)
* wdist: the distribution that regression coefficients will be sampled from
* noisedist: the distribution that noise will be sampled from
* SNR: signal-to-noise ration ||Y|| / ||noise|| we will use to normalize Y
* n: range of "number of observations" we want to test
* nfeatures: number of features
* sparsity: underlying sparsity pattern
"""
mutable struct BenchmarkData
    Xdata::XData
    wdist::BenchmarkwDists
    noisedist::BenchmarkNoiseDists
    SNR::Float64
    n::Vector{Int}
    nfeatures::Int
    sparsity::Int
end
function BenchmarkData(; Xdata::XData = XData(),
                         wdist::BenchmarkwDists = BinChoice(),
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 20.0,
                         n::Vector{Int} = [100],
                         nfeatures::Int = 100,
                         sparsity::Int = 10
                    )
    BenchmarkData(Xdata, wdist, noisedist, SNR, n, nfeatures, sparsity)
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
    # TODO smarter way to make correlated random variables. can't cache huge covariance matrices.
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
    getw(bd.nfeatures, bd.sparsity, bd.wdist)
end

function getX(n::Int, d::Int, Xdata::XData)
    rand(Xdata.dist, n, d)
end
function getX(n::Int, ::Int, Xdata::XData{T}) where {T <: MvNormal}
    rand(Xdata.dist, n)'
end
function getX(bd::BenchmarkData, i::Int)
    getX(bd.n[i], bd.nfeatures, bd.Xdata)
end

function getnoise(bnd::NoNoise, ::Float64, Y::Vector{Float64})
    n = length(Y)
    zeros(n)
end
function getnoise(bnd::BenchmarkNoiseDists, SNR::Float64, Y::Vector{Float64})
    n = length(Y)
    noise = rand(bnd, n)
    norm(noise) < 1e-3 && return noise
    (SNR < 1e-3) && error("Your SNR is too small.")
    noise .*= (norm(Y) / ( SNR * norm(noise) ) )
end
function getnoise(bd::BenchmarkData, Y::Vector{Float64})
    getnoise(bd.noisedist, bd.SNR, Y)
end
