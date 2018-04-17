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

function scalenoise!(Y::Vector{Float64}, noise::Vector{Float64}, SNR::Float64)
    norm(noise) < 1e-3 && return
    noise .*= (norm(Y) / ( sqrt(SNR) * norm(noise) ) )
    nothing
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
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 20.0,
                         n::Int = 100,
                         p::Int = 100,
                         k::Int = 10
                        )
    bd = BenchmarkData(Xdist, wdist, noisedist, SNR, n, p, k)
    w = getw(bd)
    X = getX(bd)
    noise = getnoise(bd)
    # In case we are using a multivariate distribution for x
    size(X) == (n, p) || error("X distribution doesn't match dimensions requested.")
    Y = X * w
    SNR < 1e-3 && error("Your SNR is too small.")
    scalenoise!(Y, noise, bd.SNR)
    Y .+= noise
    RegressionData(X, Y, w)
end
function getdata(bd::BenchmarkData)
    w = getw(bd)
    X = getX(bd)
    noise = getnoise(bd)
    Y = X * w
    scalenoise!(Y, noise, bd.SNR)
    Y .+= noise
    RegressionData(X, Y, w)
end
