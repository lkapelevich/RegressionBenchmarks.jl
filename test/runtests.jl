using RegressionBenchmarks
using Base.Test, Distributions

@testset "Types" begin
    srand(1)
    n = 20
    d = 10
    sparsity = 5
    Xdist = MvNormal
    Xcorr = MatrixCorrelation(0.1)
    Xdata = XData(Xdist, Xcorr, d)
    wdist = BinChoice()
    w = getw(d, sparsity, wdist)
    X = getX(n, d, Xdata)
    Y = X * w
    @test isapprox(Y[1], -1.72637, atol=1e-4)

    @test_throws ErrorException getdata(Xdata = Xdata,
        wdist = BinChoice(),
        noisedist = NoNoise(),
        n = 100, p = 100, k = 10)
    srand(1)
    rd = getdata(Xdata = Xdata, wdist = BinChoice(), n = n, p = d, k = sparsity)
    @show rd.Y[1]
    @test isapprox(rd.Y[1], -1.72637, atol=1e-4)
end
