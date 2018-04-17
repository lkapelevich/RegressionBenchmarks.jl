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
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 20.0,
                         n::Int = 100,
                         p::Int = 100,
                         k::Int = 10
                        )
    bd = BenchmarkData(Xdist, wdist, noisedist, SNR, n, p, k)
    w = getw(bd)
    X = getX(bd)
    # In case we are using a multivariate distribution for x
    size(X) == (n, p) || error("X distribution $Xdist doesn't match dimensions requested, $n by $p.")
    Y = X * w
    noise = getnoise(bd, Y)
    Y .+= noise
    RegressionData(X, Y, w)
end
function getdata(bd::BenchmarkData)
    w = getw(bd)
    X = getX(bd)
    Y = X * w
    noise = getnoise(bd, Y)
    Y .+= noise
    RegressionData(X, Y, w)
end
