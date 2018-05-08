using RegressionBenchmarks, Gurobi
using Base.Test, Distributions

srand(1)
n = [20]
d = 10
sparsity = 5
Xdist = MvNormal
Xcorr = MatrixCorrelation(0.1)
Xdata = XData(Xdist, Xcorr, d)
wdist = BinChoice()
w = getw(d, sparsity, wdist)
X = getX(n[1], d, Xdata)
Y = X * w

@testset "Types" begin
    # We are getting Y = X * w
    @test isapprox(Y[1], -1.72637, atol=1e-4)

    # We don't allow to specify X with covariance matrix of different dimensions
    # to the number of features
    @test_throws ErrorException getdata(Xdata = Xdata,
        wdist = BinChoice(),
        noisedist = NoNoise(),
        n = 100, nfeatures = 100, sparsity = 10)

    # We should also get Y = X * w with this interface
    srand(1)
    rd = getdata(Xdata = Xdata, wdist = BinChoice(), n = n[1], nfeatures = d, sparsity = sparsity)
    @test isapprox(rd.Y[1], -1.72637, atol=1e-4)

    # ... and we should also get Y = X * w with this interface
    srand(1)
    bd = BenchmarkData(Xdata = Xdata, wdist = BinChoice(),
        noisedist = NoNoise(), SNR = 0.0, n = n, nfeatures = d, sparsity = sparsity)
    rd = getdata(bd, 1)
    @test isapprox(rd.Y[1], -1.72637, atol=1e-4)

    # We don't allow stupid values for the SNR (currently 0) if user wants noise
    bd.noisedist = Normal()
    @test_throws ErrorException getdata(bd, 1)

end
@testset "Utils" begin
    @testset "Accuracy and False Alarm" begin
        pred = [1; 2; 5]
        truth = [1; 2; 3]
        a = accuracy(pred, truth)
        f = falsepositive(pred, truth)
        @test isapprox(a, 2 / 3, atol = 1e-6)
        @test isapprox(f, 1 / 3, atol = 1e-6)
    end
    @testset "Saving results" begin
        bd = BenchmarkData(Xdata = Xdata, wdist = BinChoice(),
            noisedist = NoNoise(), SNR = 0.0, n = n, nfeatures = d,
            sparsity = sparsity)
        @test data2str(bd) ==
            "x_normal_corr_rho_0.1_w_binchoice_noise_nonoise_snr_0.0_d_$(d)_k_$(sparsity)"
        m = RelaxDualSubgradient(5.0, PolyakStepping(2.0, 30), 100)
        @test method2str(m) ==
            "relax_dual_subgradient_polyak_2.0_maxiter_100"
        m = ExactPrimalCuttingPlane(0.1, 30.0, GurobiSolver)
        @test method2str(m) ==
            "exact_primal_tlimit_30.0_gurobi"
    end
end

@testset "Methods" begin
    srand(1)
    bd = BenchmarkData(Xdata = Xdata, wdist = BinChoice(),
        noisedist = NoNoise(), SNR = 0.0, n = n, nfeatures = d, sparsity = sparsity)
    rd = getdata(bd, 1)
    @testset "Subgradient" begin
        @testset "Ordinary" begin
            m = RelaxDualSubgradient(5.0, ConstantStepping(1e-3), 100)
            indices, weights = solve_problem(m, rd.X, rd.Y, sparsity)
            @test indices == find(abs.(w) .> 1e-6)
            @test isapprox(weights, w[abs.(w) .> 1e-6], atol = 1e-1)
        end
        @testset "Varying stepsizes" begin
            m = RelaxDualSubgradient(50.0, PolyakStepping(0.5, 30), 100)
            indices, weights = solve_problem(m, rd.X, rd.Y, sparsity)
            @test indices == find(abs.(w) .> 1e-6)
            @test isapprox(weights, w[abs.(w) .> 1e-6], atol = 1e-1)
        end
    end
end

@testset "Validation" begin
    srand(1)
    bd = BenchmarkData(Xdata = Xdata, wdist = BinChoice(),
        noisedist = NoNoise(), SNR = 0.0, n = n, nfeatures = d, sparsity = sparsity)
    rd = getdata(bd, 1)
    for method in [ExactPrimalCuttingPlane(GurobiSolver);
                   PrimalWithHeuristics(GurobiSolver);
                   RelaxDualSubgradient();
                   RelaxDualSubgradient(50.0, PolyakStepping(), 100)
                  ]
        v_results = validate_params!(rd.X, rd.Y, bd.sparsity, method)
        @test any(isapprox.(v_results.valid_scores, 1.0, atol=1e-2))
        if isa(method, Union{ExactPrimalCuttingPlane, PrimalWithHeuristics})
            @test method.time_limit == 30.0
        end
    end
end
