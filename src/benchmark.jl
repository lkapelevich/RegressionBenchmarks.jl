# reload("RegressionBenchmarks")
using RegressionBenchmarks, Distributions, MLDataUtils

"""
    accuracy(pred::Vector{Int}, truth::Vector{Int})

Returns proportion of indices in `truth` that are also in `pred`.
"""
function accuracy(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for t in truth
        (t in pred) && (detected += 1)
    end
    detected / length(truth)
end

"""
    falsepositive(pred::Vector{Int}, truth::Vector{Int})

Returns the proportion of indices in `pred` that are not in `truth`.
"""
function falsepositive(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for p in pred
        (p in truth) || (detected += 1)
    end
    detected / length(pred)
end

function benchmark(bd::BenchmarkData, method::RegressionMethod)

    # Generate some synthetic data
    rd = getdata(bd)
    true_support = find(rd.w > 0)
    results = [0.0, 0.0, 0.0, 0.0]

    folds = kfolds((bd.X, bd.Y), k = 5)
    for ((X_train, y_train), (X_test, y_test)) in folds
        # Validate the right method parameters
        validate_params!(X_train, y_train, rd, method)
        tic()
        indices, w = solve_problem(data, method)
        time = toq()
        a = accuracy(indices, true_support)
        f = false_positive(indices, true_support)
        mse = sum(abs2.(rd.Y - bd.X * w))
        results = [a, f, mse, time]
    end
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
bd = BenchmarkData(MvNormal(μ * ones(d), Σ), BinChoice(), n, d, sparsity)
# Validate
# Test
m = ExactPrimalCuttingPlane()
results = benchmark_notest(bd, m)
