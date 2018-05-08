# This file constains code copied from https://github.com/jeanpauphilet/SubsetSelection.jl
# as at commit e6b3258ba394f6152853498d81d3cecb9d67e097

const sgtol = 1e-6

mutable struct PolyakCache
  best_inds::Vector{Int}
  best_upper::Float64
  best_lower::Float64
end
PolyakCache(inds::Vector{Int}) = PolyakCache(inds, Inf, -Inf)

abstract type SteppingRule end
struct ConstantStepping <: SteppingRule
  stepsize::Float64
end
mutable struct PolyakStepping <: SteppingRule
  initial_factor::Float64        # initial factor for scaling (ub-lb)/norm^2
  wait::Int64                    # currently unused TODO
  function PolyakStepping(initial_factor, wait)
      if 0.0 <= initial_factor <= 2.0
          new(initial_factor, wait)
      else
          error("Initial stepping parameter must be between 0 and 2. Current
                    choice is $initial_factor.")
      end
  end
end
PolyakStepping() = PolyakStepping(2.0, 30)

function resetindices(::PolyakStepping)
  true
end
function resetindices(::SteppingRule)
  false
end

function getdelta!(sr::SteppingRule, ::Any, ::Array{Float64,2}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int}, ::Int, ::Float64, ::SubsetSelection.Cache, ::PolyakCache)
  error("Need to define `getdelta` for stepping rule $(sr).")
end
function getdelta!(sr::ConstantStepping, ::Any, ::Array{Float64,2}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int}, ::Int, ::Float64, ::SubsetSelection.Cache, ::PolyakCache)
  sr.stepsize
end
function getdelta!(sr::PolyakStepping,
      Y::Union{Vector{Float64},SubArray{Float64}}, X::Array{Float64,2},
      α::Vector{Float64}, ∇::Vector{Float64},
      indices::Vector{Int}, n_indices::Int, γ::Float64,
      cache::SubsetSelection.Cache, pc::PolyakCache
     )
  lower_bound = dual_bound(SubsetSelection.OLS(), Y, X, α, indices, n_indices, γ, cache)
  upper_bound = primal_bound(SubsetSelection.OLS(), Y, X, γ, indices, n_indices)
  if upper_bound < pc.best_upper
    pc.best_upper = upper_bound
    pc.best_inds .= indices
  end
  (lower_bound > pc.best_lower) && (pc.best_lower = lower_bound)
  if lower_bound - sgtol > upper_bound + sgtol
      error("Bounds overlap. LB = $lowerbound and UB = $upper_bound.")
  end
  sr.initial_factor * (pc.best_upper - lower_bound) / sum(abs2.(∇))
end

##############################################
##DUAL SUB-GRADIENT ALGORITHM
##############################################
""" Function to compute a sparse regressor/classifier. Solve an optimization problem of the form
        min_s max_α f(α, s)
by gradient ascent on α:        α_{t+1} ← α_t + δ ∇_α f(α_t, s_t)
and partial minimization on s:  s_{t+1} ← argmin_s f(α_t, s)
INPUTS
- ℓ           Loss function used
- Card        Model to enforce sparsity (constraint or penalty)
- Y (n×1)     Vector of outputs. For classification, use ±1 labels
- X (n×p)     Array of inputs.
- indInit     (optional) Initial subset of features s
- αInit       (optional) Initial dual variable α
- γ           (optional) ℓ2 regularization penalty
- intercept   (optional) Boolean. If true, an intercept term is computed as well
- maxIter     (optional) Total number of Iterations
- δ           (optional) Gradient stepsize
- gradUp      (optional) Number of gradient updates of dual variable α performed per update of primal variable s
- anticycling (optional) Boolean. If true, the algorithm stops as soon as the support is not unchanged from one iteration to another
- averaging   (optional) Boolean. If true, the dual solution is averaged over past iterates
OUTPUT
- SparseEstimator """
function subsetSelection_bm(ℓ::LossFunction, Card::Sparsity,
    Y::Union{Vector{Float64},SubArray{Float64}},
    X::Array{Float64,2};
    indInit = SubsetSelection.ind_init(Card, size(X,2)),
    αInit = SubsetSelection.alpha_init(ℓ, Y),
    γ = 1/sqrt(size(X,1)),  intercept = false,
    maxIter = 100, sr::SteppingRule = ConstantStepping(1e-3), gradUp = 1,
    anticycling = false, averaging = true)

  n,p = size(X)
  cache = SubsetSelection.SubsetSelection.Cache(n, p)

  #Add sanity checks
  if size(Y,1) != n
    throw(DimensionMismatch("X and Y must have the same number of rows"))
  end
  if isa(ℓ, SubsetSelection.Classification)
    levels = sort(unique(Y))
    if length(levels) != 2
      throw(ArgumentError("subsetSelection only supports two-class classification"))
    elseif (levels[1] != -1) || (levels[2] != 1)
      throw(ArgumentError("Class labels must be ±1's"))
    end
  end


  indices = indInit #Support
  n_indices = length(indices)
  best_inds = copy(indices)

  n_indices_max = SubsetSelection.max_index_size(Card, p)
  resize!(indices, n_indices_max)

  indices_old = Vector{Int}(n_indices_max)
  α = αInit[:]  #Dual variable α
  a = αInit[:]  #Past average of α

  # We will compute the dual bound incorrectly unless indicies match initial α
  if resetindices(sr)
    n_indices = SubsetSelection.partial_min!(indices, Card, X, α, γ, cache)
  end

  pc = PolyakCache(copy(indices))

  bounds_match = false

  ##Dual Sub-gradient Algorithm
  for iter in 2:maxIter

    #Gradient ascent on α
    for _ in 1:min(gradUp, div(p, n_indices))
      ∇ = SubsetSelection.grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache)
      δ = getdelta!(sr, Y, X, α, ∇, indices, n_indices, γ, cache, pc)
      α .+= δ*∇
      α = SubsetSelection.proj_dual(ℓ, Y, α)
      α = SubsetSelection.proj_intercept(intercept, α)
    end

    if !all(isfinite.(α))
        warn("Algorithm diverges! Did you normalize your data? Otherwise, try reducing stepsize δ.")
    end
    #Update average a
    @__dot__ a = (iter - 1) / iter * a + 1 / iter * α
    # a *= (iter - 1)/iter; a .+= α/iter

    #Minimization w.r.t. s
    indices_old[1:n_indices] = indices[1:n_indices]
    n_indices = SubsetSelection.partial_min!(indices, Card, X, α, γ, cache)

    #Anticycling rule: Stop if indices_old == indices
    if anticycling && SubsetSelection.indices_same(indices, indices_old, n_indices)
      averaging = false #If the algorithm stops because of cycling, averaging is not needed
      break
    end
    # If lower and upper bounds have closed, skip to the end
    if abs(pc.best_upper - pc.best_lower) / (1 + abs(pc.best_upper)) < 1e-4
      indices .= pc.best_inds
      break
    end
  end

  ##Compute sparse estimator
  #Subset of relevant features
  n_indices = SubsetSelection.partial_min!(indices, Card, X, averaging ? a : α, γ, cache)
  #Regressor
  # w = [-γ * dot(X[:, indices[j]], a) for j in 1:n_indices]
  w = SubsetSelection.recover_primal(ℓ, Y, X[:,indices], γ)
  #Bias
  b = SubsetSelection.compute_bias(ℓ, Y, X, a, indices, n_indices, γ, intercept, cache)

  #Resize final indices vector to only have relevant entries
  resize!(indices, n_indices)

  return SubsetSelection.SparseEstimator(ℓ, Card, indices, w, a, b, maxIter)
end

function ax_squared(X, α::Vector{Float64}, indices::Vector{Int}, n_indices::Int)
  # TODO update ax in cache and use it rather than recomputing
  axsum = 0.0
  for j = 1:n_indices
    x = @view(X[:, indices[j]])
    axsum += dot(α, x)^2
  end
  axsum
end

function primal_bound(ℓ::SubsetSelection.OLS, Y::Union{Vector{Float64},SubArray{Float64}}, X::Array{Float64,2}, γ::Float64, indices::Vector{Int}, n_indices::Int)
  αstar = SubsetSelectionCIO.sparse_inverse(ℓ, Y, X[:, indices], γ) # TODO could do this less often. also don't return vector after function.
  axsum = ax_squared(X, αstar, indices, n_indices)
  bound = -0.5 * dot(αstar, αstar) - dot(Y, αstar) - γ * 0.5 * axsum
  # Normalize TODO normalize this and grad dual
  # bound / size(X, 1)
end

function dual_bound(ℓ::SubsetSelection.OLS, Y::Union{Vector{Float64},SubArray{Float64}}, X::Array{Float64,2}, α::Vector{Float64}, indices::Vector{Int}, n_indices::Int, γ, cache::SubsetSelection.Cache)
  axsum = ax_squared(X, α, indices, n_indices)
  bound = -0.5 * dot(α, α) - dot(Y, α) - γ * 0.5 * axsum
  # Normalize
  # bound / size(X, 1)
end
