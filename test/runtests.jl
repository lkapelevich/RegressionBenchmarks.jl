using RegressionBenchmarks
using Base.Test, Distributions

@testset "Types" begin
    srand(1)
    n = 20
    d = 10
    sparsity = 5
    μ = 0.0
    Σ = 0.1 * ones(d, d)
    @inbounds for i = 1:d
        Σ[i, i] = 1.0
    end
    Xdist = MvNormal(μ * ones(d), Σ)
    wdist = BinChoice()
    w = getw(d, sparsity, wdist)
    X = getX(n, d, Xdist)
    Y = X * w
    @test isapprox(Y[1], -1.72637, atol=1e-4)

    @test_throws ErrorException getdata(Xdist = MvNormal(μ * ones(d), Σ), wdist = BinChoice(), n = 100, p = 100, k = 10)
    srand(1)
    bd = getdata(Xdist = MvNormal(μ * ones(d), Σ), wdist = BinChoice(), n = n, p = d, k = sparsity)
    @test isapprox(bd.Y[1], -1.72637, atol=1e-4)
end
