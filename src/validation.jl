function validate_params!(bd::BenchmarkData, m::RegressionMethod)
    error("No validation method for $m.")
end

function validate_params!(bd::BenchmarkData, rd::RegressionData, m::ExactPrimalCuttingPlane)

    # Just need to validate gamma
    nfolds = 10
    folds = kfolds(rd.X, rd.Y, k = nfolds)

    gamma_factors = 2.^collect(0:10)

    best_gammas = zeros(nfolds)

    for ((X, Y), i) in enumerate(folds)
        best_score = Inf
        n = size(X, 2)
        m.gamma = 1 / sqrt(n)
        for f in gamma_factors
            m.gamma /= f
            score = solve(X, Y, m)
            if score > best_score
                best_score = score
                best_gammas[i] = m.gamma
            end
        end
    end
    m.gamma = mean(best_gammas)
end
