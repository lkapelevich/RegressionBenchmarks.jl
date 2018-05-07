mutable struct ValidationResults{T <: RegressionMethod}
    m_with_params::Vector{T}
    train_scores::Vector{Float64}
    valid_scores::Vector{Float64}
end
ValidationResults(m::RegressionMethod) = ValidationResults(m, Float64[], Float64[])

function validate_params!(bd::BenchmarkData, m::RegressionMethod)
    error("No validation method for $m.")
end

function settimelimit!(m::Union{ExactPrimalCuttingPlane, PrimalWithHeuristics}, t::Float64)
    old_tl = m.time_limit
    m.time_limit = t
    old_tl
end
function settimelimit!(::RelaxDualSubgradient, ::Float64)
    0.0
end

const GammaMethods = Union{ExactPrimalCuttingPlane, PrimalWithHeuristics, RelaxDualSubgradient}

function validate_stepsize!(::RegressionMethod,
                                ::Array{Float64,2},
                                ::Union{Vector{Float64},SubArray{Float64}},
                                ::Array{Float64,2},
                                ::Union{Vector{Float64},SubArray{Float64}},
                                ::Int
                            )
    nothing
end

function validate_stepsize!(m::RelaxDualSubgradient{SR},
                                X_train::Array{Float64,2},
                                Y_train::Union{Vector{Float64},SubArray{Float64}},
                                X_valid::Array{Float64,2},
                                Y_valid::Union{Vector{Float64},SubArray{Float64}},
                                sparsity::Int
                            ) where {SR <: PolyakStepping}

    best_score = -Inf
    best_if = 0.5
    ifrange = [0.5, 1.0, 2.0]
    for initial_factor in ifrange
        m.sr.initial_factor = initial_factor
        indices, w = solve_problem(m, X_train, Y_train, sparsity)
        pred = predict_sparse(X_valid, indices, w)
        valid_score = oosRsquared(pred, Y_valid, Y_train)
        if valid_score > best_score
            best_score = valid_score
            best_if = initial_factor
        end
    end
    m.sr.initial_factor = best_if
    nothing
end

"""
    validate_params!(X::Array{Float64,2},
                    Y::Union{Vector{Float64},SubArray{Float64}},
                    sparsity::Int,
                    m::GammaMethods)

Just need to validate gamma.
"""
function validate_params!(X::Array{Float64,2},
                        Y::Union{Vector{Float64},SubArray{Float64}},
                        sparsity::Int,
                        m::GammaMethods)

    nfolds = 10
    folds = kfolds(shuffleobs((X', Y)), k = nfolds)

    # Range of gammas we are going to validate # TODO make input
    n = size(X, 1)
    gamma_range = 1 / sqrt(n) .* 2.^collect(0:10)
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

    old_tl = settimelimit!(m, 60.0)

    # Validate over all folds in the data
    fold = 1
    for ((X_train, Y_train), (X_valid, Y_valid)) in folds
        X_train, X_valid = X_train', X_valid'
        best_score = -Inf
        for (i, f) in enumerate(gamma_range)
            m.gamma = f
            # If relevant, find the best scaling factor for varying stepsizes
            validate_stepsize!(m, X_train, Y_train, X_valid, Y_valid, sparsity)
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
    settimelimit!(m, old_tl)
    v_results.train_scores ./= nfolds
    v_results.valid_scores ./= nfolds
    m.gamma = mean(best_gammas)
    v_results
end

function valid2io(io::IO, vresults::Array{ValidationResults,2}, nrange::Vector{Int})
    @assert length(nrange) == size(vresults, 1)
    write(io, "test_fold, n, gamma, mean_train_score, mean_valid_score \n")
    for j = 1:size(vresults, 2)
        for i = 1:length(nrange)
            for k = 1:length(vresults[i, j].m_with_params)
              write(io, "$j, $(nrange[i]), $(vresults[i, j].m_with_params[k].gamma), $(vresults[i, j].train_scores[k]), $(vresults[i, j].valid_scores[k]) \n")
            end
        end
    end
end
