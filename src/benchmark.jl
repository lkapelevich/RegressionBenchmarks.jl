function benchmark(bd::BenchmarkData, method::RegressionMethod, nfolds::Int=5)

    nrange = length(bd.n)
    results = zeros(nrange, 6)

    validation_results = Array{ValidationResults,2}(nrange, nfolds)

    for i = 1:nrange
        # Generate some synthetic data
        rd = getdata(bd, i)
        # Normalize it
        # normalize!(rd)
        true_support = find(abs.(rd.w) .> 1e-6)

        folds = kfolds(shuffleobs((rd.X', rd.Y)), k = nfolds)
        j = 0
        for ((X_train, y_train), (X_test, y_test)) in folds
            j += 1
            # Validate the right method parameters
            X_train, X_test = X_train', X_test'
            v_results = validate_params!(X_train, y_train, bd.sparsity, method)
            validation_results[i, j] = v_results
            tic()
            indices, w = solve_problem(method, X_train, y_train, bd.sparsity)
            time = toq()
            a = accuracy(indices, true_support)
            f = falsepositive(indices, true_support)
            pred_train = predict_sparse(X_train, indices, w)
            train_R2 = isRsquared(pred_train, y_train)
            pred_test  = predict_sparse(X_test, indices, w)
            test_R2 = oosRsquared(pred_test, y_test, y_train)
            @show bd.n[i], a
            results[i, :] .+= [a, f, train_R2, test_R2, time, method.gamma]
        end
    end
    results / nfolds, validation_results
end
