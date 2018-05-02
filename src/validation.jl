mutable struct ValidationResults{T <: RegressionMethod}
    m_with_params::Vector{T}
    train_scores::Vector{Float64}
    valid_scores::Vector{Float64}
end

function validate_params!(bd::BenchmarkData, m::RegressionMethod)
    error("No validation method for $m.")
end

const GammaMethods = Union{ExactPrimalCuttingPlane, PrimalWithHeuristics}

"""
    validate_params!(X::Array{Float64,2}, Y, sparsity::Int, m::GammaMethods)

Just need to validate gamma.
"""
function validate_params!(X::Array{Float64,2}, Y, sparsity::Int, m::GammaMethods)

    nfolds = 10
    folds = kfolds(shuffleobs((X', Y)), k = nfolds)

    # Range of gammas we are going to validate # TODO make input
    n = size(X, 2)
    gamma_range = sqrt(n) ./ 2.^collect(0:10)
    nvals = length(gamma_range)

    # Cache the methods with what we are validating in case we want to save to
    # file validation results later
    methods_cache = Vector{ExactPrimalCuttingPlane}(nvals)
    for (i, g) in enumerate(gamma_range)
        methods_cache[i] = ExactPrimalCuttingPlane(g, 30.0)
    end

    # Cache results from each gamma in case it is of interest
    v_results = ValidationResults(methods_cache, zeros(nvals), zeros(nvals))

    # Gammas giving highest validation score in each fold
    best_gammas = zeros(nfolds)

    # Validate over all folds in the data
    fold = 1
    for ((X_train, Y_train), (X_valid, Y_valid)) in folds
        X_train, X_valid = X_train', X_valid'
        best_score = -Inf
        for (i, f) in enumerate(gamma_range)
            m.gamma = f
            # Solve the problem
            indices, w = solve_problem(m, X_train, Y_train, sparsity)
            # Make in sample and out of sample predictions
            pred = predict_sparse(X_valid, indices, w)
            pred_train = predict_sparse(X_train, indices, w)
            # Get scores
            train_score = isRsquared(pred_train, Y_train)
            valid_score = oosRsquared(pred, Y_valid, Y_train)
            # Cache scores in case they are of interest
            v_results.train_scores[i] += train_score
            v_results.valid_scores[i] += valid_score
            # Update best gamma and score if relevant
            if valid_score > best_score
                best_score = valid_score
                best_gammas[fold] = m.gamma
            end
        end
        fold += 1
    end
    v_results.train_scores ./= nfolds
    v_results.valid_scores ./= nfolds
    m.gamma = mean(best_gammas)
    v_results
end
