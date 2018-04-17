# reload("RegressionBenchmarks")
using RegressionBenchmarks, Distributions, MLDataUtils

function benchmark(bd::BenchmarkData, method::RegressionMethod)

    # Generate some synthetic data
    rd = getdata(bd)
    true_support = find(rd.w .> 0)
    results = [0.0, 0.0, 0.0, 0.0, 0.0]

    folds = kfolds(shuffleobs((rd.X', rd.Y)), k = 5)
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
        results = [a, f, train_R2, test_R2, time]
    end
    results
end



srand(1)
n = 20
d = 10
sparsity = 5
μ = 0.0
Σ = 0.1 * ones(d, d)
@inbounds for i = 1:d
    Σ[i, i] = 1.0
end

# Get data part
bd = BenchmarkData(MvNormal(μ * ones(d), Σ), BinChoice(), NoNoise(), 0.0, n, d, sparsity)
# Validate
# Test
m = ExactPrimalCuttingPlane()
results = benchmark(bd, m)
