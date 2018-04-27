# reload("RegressionBenchmarks")
using RegressionBenchmarks, Distributions, MLDataUtils

function benchmark(bd::BenchmarkData, method::RegressionMethod)

    # Generate some synthetic data
    rd = getdata(bd)
    # Normalize it
    normalize!(rd)
    true_support = find(rd.w .> 0)
    results = [0.0, 0.0, 0.0, 0.0, 0.0]

    nfolds = 5

    folds = kfolds(shuffleobs((rd.X', rd.Y)), k = nfolds)
    for ((X_train, y_train), (X_test, y_test)) in folds
        # Validate the right method parameters
        X_train, X_test = X_train', X_test'
        validate_params!(X_train, y_train, bd.k, method)
        tic()
        indices, w = solve_problem(method, X_train, y_train, bd.k)
        time = toq()
        a = accuracy(indices, true_support)
        f = falsepositive(indices, true_support)
        pred_train = predict_sparse(X_train, indices, w)
        train_R2 = isRsquared(pred_train, y_train)
        pred_test  = predict_sparse(X_test, indices, w)
        test_R2 = oosRsquared(pred_test, y_test, y_train)
        results .+= [a, f, train_R2, test_R2, time]
    end
    results / nfolds
end


nrange = 100:100:1000
results_table = Array{Float64,2}(length(nrange), 5)
d = 10
srand(1)

for (ni, n) in enumerate(nrange)
    sparsity = 5
    Xdist = MvNormal
    Xcorr = MatrixCorrelation(0.1)
    Xdata = XData(Xdist, Xcorr, d)
    # Get data part
    bd = BenchmarkData(Xdata, BinChoice(), NoNoise(), 0.0, n, d, sparsity)
    # Validate
    # Test
    m = ExactPrimalCuttingPlane()

    results_table[ni, :] .= benchmark(bd, m)
end
