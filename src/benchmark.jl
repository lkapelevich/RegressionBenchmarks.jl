# reload("RegressionBenchmarks")
using RegressionBenchmarks, Distributions, MLDataUtils

function benchmark(bd::BenchmarkData, method::RegressionMethod)

    nrange = length(bd.n)
    results = zeros(nrange, 6)
    nfolds = 5

    for i = 1:nrange
        # Generate some synthetic data
        rd = getdata(bd, i)
        # Normalize it
        # normalize!(rd)
        true_support = find(abs.(rd.w) .> 1e-6)

        folds = kfolds(shuffleobs((rd.X', rd.Y)), k = nfolds)
        for ((X_train, y_train), (X_test, y_test)) in folds
            # Validate the right method parameters
            X_train, X_test = X_train', X_test'
            validate_params!(X_train, y_train, bd.sparsity, method)
            tic()
            indices, w = solve_problem(method, X_train, y_train, bd.sparsity)
            time = toq()
            a = accuracy(indices, true_support)
            f = falsepositive(indices, true_support)
            pred_train = predict_sparse(X_train, indices, w)
            train_R2 = isRsquared(pred_train, y_train)
            pred_test  = predict_sparse(X_test, indices, w)
            test_R2 = oosRsquared(pred_test, y_test, y_train)
            results[i, :] .+= [a, f, train_R2, test_R2, time, method.gamma]
        end
    end
    results / nfolds
end

function makeplot(field::String)
    p = plot(df, x = nrange, y = field)
    draw(PNG(joinpath(datadir,
                    "$(field)_sparsity$(sparsity).png"
            ), 3inch, 3inch), p)
    nothing
end

nrange = collect(100:100:1000)
d = 10
srand(1)
Xdist = MvNormal
Xcorr = MatrixCorrelation(0.1)
Xdata = XData(Xdist, Xcorr, d)
sparsity = 5

# Get data part
bd = BenchmarkData(Xdata, BinChoice(), NoNoise(), 0.0, nrange, d, sparsity)
# Choose model
m = ExactPrimalCuttingPlane()
# Get results
results_table .= benchmark(bd, m)
# Save
!isdir(datadir) && mkdir(datadir)
df = convert(DataFrame, results_table)
names!(df, [:accuracy, :false, :train_r2, :test_r2, :time, :gamma])
writetable(joinpath(datadir, "sparsity$(sparsity).csv"), df)
# Make plots
for field in ["accuracy", "false", "test_r2", "time"]
    makeplot(field)
end
