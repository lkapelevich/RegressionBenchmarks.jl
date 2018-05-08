function initial_ub(X::Array{Float64,2}, Y::Union{Vector{Float64},SubArray{Float64}},
                    s0::Vector{Float64}, sparsity::Int, γ::Float64)
    αstar = SubsetSelectionCIO.sparse_inverse(OLS(), Y, X[:, 1:sparsity], γ)
    axsum = ax_squared(X, αstar, collect(1:sparsity), sparsity)
    -0.5 * dot(αstar, αstar) - dot(Y, αstar) - γ * 0.5 * axsum
end

function getlb(Y::Union{Vector{Float64},SubArray{Float64}}, α::Vector{Float64}, ax_sparse::Vector{Float64})
    -0.5 * dot(α, α) + dot(Y, α) - γ * 0.5 * sum(ax_sparse.^2)
end

function getslope!(∇::Vector{Float64}, X::Array{Float64,2},
            Y::Union{Vector{Float64},SubArray{Float64}}, α::Vector{Float64},
            indices::Vector{Int}, γ::Float64, sparsity::Int,
            ax_sparse::Vector{Float64})
    ∇ .= -α + Y
    g = 0.0
    @inbounds for j = 1:sparsity
        x = @view(X[:, indices[j]])
        g .-= ax_sparse[j] * x
    end
    ∇ .-= γ * g
    nothing
end

function solve_dualcutting(X::Array{Float64,2},
            Y::Union{Vector{Float64},SubArray{Float64}},
            sparsity::Int,
            γ::Float64,
            solver::MathProgBase.MathProgSolver)

    n, p = size(X)
    @assert sparsity <= p
    indices = collect(1:sparsity)
    initial_bound = initial_ub(X, Y, s0, sparsity, γ)

    ax = Vector{Float64}(n)
    ∇  = Vector{Float64}(n)
    ax_sparse = Vector{Float64}(sparsity)

    ub = initial_bound
    lb = -Inf

    m = Model(solver = solver)
    @variables(m, begin
        α
        η <= initial_bound
    end)
    @objective(m, Max, η)

    while (ub - lb) / (1 + abs(ub)) <= 1e-6
        # Solve cutting plane model
        @assert solve(m) == :Optimal
        # Upper bound is current optimum
        ub = getobjectivevalue(m)
        # Get the α that attains the optimum
        α_hat = getvalue(m[:α])
        # Recover a primal solution
        ax = X * α_hat # TODO don't need to store all at the same time
        indices = sortperm(-ax)[1:sparsity]
        X_sparse = @view(X[:, indices])
        ax_sparse = X_sparse' * a
        # Use it to update gradient
        getslope!(∇, X, Y, α_hat, indices, γ, sparsity, ax_sparse)
        # Lower bound is the actual function value at α_hat
        lb = getlb(Y, α_hat, ax_sparse)
        # Add a cut
        @constraint(m, η >= lb + dot(∇, α - α_hat))
    end
    # Recover weights
    w = SubsetSelection.recover_primal(OLS(), Y, X[:,indices], γ)
    indices, w
end
