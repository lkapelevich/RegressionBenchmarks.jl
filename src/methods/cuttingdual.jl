function initial_ub(X::Array{Float64,2}, Y::Union{Vector{Float64},SubArray{Float64}},
                    s0::Vector{Float64}, sparsity::Int, γ::Float64)
    αstar = SubsetSelectionCIO.sparse_inverse(OLS(), Y, X[:, 1:sparsity], γ)
    axsum = ax_squared(X, αstar, collect(1:sparsity), sparsity)
    -0.5 * dot(αstar, αstar) - dot(Y, αstar) - γ * 0.5 * axsum
end

function solve_dualcutting(X::Array{Float64,2},
            Y::Union{Vector{Float64},SubArray{Float64}},
            sparsity::Int,
            γ::Float64)

    n, p = size(X)
    s0 = zeros(p)
    s0[1:sparsity] .= 1.0
    initial_bound = initial_ub(X, Y, s0, sparsity, γ)
end
