mutable struct RegressionData
    X::Array{Float64,2}
    Y::Vector{Float64}
    w::Vector{Float64}
end


"""
    getdata(; Xdata::XData = XData(),
                         wdist::BenchmarkwDists = BinChoice(),
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 0.0,
                         n::Int = 100,
                         nfeatures::Int = 100,
                         sparsity::Int = 10
                )

Get regression data. Created to be stand-alone from the rest of the
benchmarking. Maybe will remove later, want user to only interface with
BenchmarkData object.
"""
function getdata(; Xdata::XData = XData(),
                         wdist::BenchmarkwDists = BinChoice(),
                         noisedist::BenchmarkNoiseDists = NoNoise(),
                         SNR::Float64 = 0.0,
                         n::Int = 100,
                         nfeatures::Int = 100,
                         sparsity::Int = 10
                        )
    bd = BenchmarkData(Xdata, wdist, noisedist, SNR, [n], nfeatures, sparsity)
    getdata(bd, 1)
end
"""
    getdata(bd::BenchmarkData, i::Int)

Get regression data, the i^th value in the range for n will be used.
"""
function getdata(bd::BenchmarkData, i::Int)
    w = getw(bd)
    X = getX(bd, i)
    # In case we are using a multivariate distribution for x
    if size(X, 2) != (bd.nfeatures)
        error("X distribution $(bd.Xdata.dist) doesn't match the number of
                features requested, $(bd.nfeatures).")
    end
    Y = X * w
    noise = getnoise(bd, Y)
    Y .+= noise
    RegressionData(X, Y, w)
end

function Base.normalize!(rd::RegressionData)
    for p = 1:size(rd.X, 2)
        μ = mean(rd.X[:, p])
        σ = std(rd.X[:, p])
        (σ < 1e-6) && continue
        rd.X[:, p] .= (rd.X[:, p] - μ) / σ
    end
    rd.Y .= (rd.Y - mean(rd.Y)) / std(rd.Y)
    nothing
end
