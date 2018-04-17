function validate_params!(bd::BenchmarkData, m::RegressionMethod)
    error("No validation method for $m.")
end

function validate_params!(X::Array{Float64,2}, Y, sparsity::Int, m::ExactPrimalCuttingPlane)

    # Just need to validate gamma
    nfolds = 10
    folds = kfolds(shuffleobs((X', Y)), k = nfolds)

    gamma_factors = 2.^collect(0:10)

    best_gammas = zeros(nfolds)

    fold = 1
    for ((X_train, Y_train), (X_valid, Y_valid)) in folds
        X_train, X_valid = X_train', X_valid'
        best_score = -Inf
        n = size(X_train, 2)
        m.gamma = sqrt(n)
        for f in gamma_factors
            m.gamma /= f
            indices, w = solve_problem(m, X_train, Y_train, sparsity)
            pred = predict_sparse(X_valid, indices, w)
            score = oosRsquared(pred, Y_valid, Y_train)
            # @show score
            if score > best_score
                best_score = score
                best_gammas[fold] = m.gamma
            end
        end
        fold += 1
    end
    # @show best_gammas
    m.gamma = mean(best_gammas)
end
