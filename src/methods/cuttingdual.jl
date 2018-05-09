function initial_ub(X::Array{Float64,2}, Y::YVector, sparsity::Int, γ::Float64)
    αstar = -SubsetSelectionCIO.sparse_inverse(OLS(), Y, X[:, 1:sparsity], γ)
    X_sparse = @view(X[:, 1:sparsity])
    ax_sparse = X_sparse' * αstar
    getlb(Y, αstar, ax_sparse, γ)
end

function getlb(Y::YVector, α::Vector{Float64}, ax_sparse::Vector{Float64}, γ::Float64)
    -0.5 * dot(α, α) + dot(Y, α) - γ * 0.5 * sum(ax_sparse.^2)
end

function getslope!(∇::Vector{Float64}, X::Array{Float64,2},
            Y::YVector, α::Vector{Float64},
            indices::Vector{Int}, γ::Float64, sparsity::Int,
            ax_sparse::Vector{Float64})
    ∇ .= -α + Y
    g = zeros(length(α))
    @inbounds for j = 1:sparsity
        x = @view(X[:, indices[j]])
        g .+= ax_sparse[j] * x
    end
    ∇ .-= γ * g
    nothing
end

"""
    solve_dualcutting(X::Array{Float64,2},
            Y::YVector,
            sparsity::Int,
            γ::Float64,
            solver::MathProgBase.AbstractMathProgSolver)
"""
function solve_dualcutting(X::Array{Float64,2},
            Y::YVector,
            sparsity::Int,
            γ::Float64,
            solver::MathProgBase.AbstractMathProgSolver)

    maxiter = 10_0000

    n, p = size(X)
    @assert sparsity <= p
    indices = collect(1:sparsity)
    initial_bound = initial_ub(X, Y, sparsity, γ)

    ax = Vector{Float64}(p)
    ∇  = Vector{Float64}(n)
    ax_sparse = Vector{Float64}(sparsity)
    α_hat = Vector{Float64}(n)

    ub = initial_bound
    lb = -Inf

    m = Model(solver = solver)
    @variables(m, begin
        α[1:n]
        0 <= η <= initial_bound
    end)
    @objective(m, Max, η)

    iter = 0

    while (ub - lb) / (1 + abs(ub)) > 1e-4
        # Solve cutting plane model
        @assert solve(m) == :Optimal
        # Upper bound is current optimum
        ub = getobjectivevalue(m)
        # Get the α that attains the optimum
        α_hat .= getvalue(m[:α])
        # Recover a primal solution
        ax .= X' * α_hat # TODO don't need to store all at the same time
        indices .= sortperm(-abs.(ax))[1:sparsity]
        X_sparse = @view(X[:, indices])
        ax_sparse .= X_sparse' * α_hat        # Use it to update gradient
        getslope!(∇, X, Y, α_hat, indices, γ, sparsity, ax_sparse)
        # Lower bound is the actual function value at α_hat
        lb = getlb(Y, α_hat, ax_sparse, γ)
        # Add a cut
        @constraint(m, η <= lb + dot(∇, α - α_hat))
        if iter > maxiter
            break
        end
    end
    # Recover weights
    w = SubsetSelection.recover_primal(OLS(), Y, X[:,indices], γ)
    perm = sortperm(indices)
    indices[perm], w[perm]
end
