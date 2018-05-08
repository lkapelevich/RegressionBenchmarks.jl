# This file was copied from https://github.com/jeanpauphilet/SubsetSelectionCIO.jl
# as at commit a21d34652e0349a6b6f33e9c926ffad659d05e26

struct UnsetSolver <: MathProgBase.AbstractMathProgSolver end
function getsolver(s::Type{S}, tl::Float64) where {S <: MathProgBase.AbstractMathProgSolver}
    UnsetSolver()
end
function getsolver(s::Type{S}, tl::Float64) where {S <: GurobiSolver}
    GurobiSolver(OutputFlag = 0, TimeLimit = tl)
end
function getsolver(s::Type{S}, tl::Float64) where {S <: CplexSolver}
    CplexSolver(CPX_PARAM_SCRIND = 0, CPX_PARAM_MIPDISPLAY = 0,
                CPX_PARAM_DETTILIM = tl)
end

###########################
# FUNCTION oa_formulation
###########################
"""Computes the minimum regression error with Ridge regularization subject an explicit
cardinality constraint using cutting-planes.

w^* := arg min  ∑_i ℓ(y_i, x_i^T w) +1/(2γ) ||w||^2
           st.  ||w||_0 = k

INPUTS
  ℓ           - LossFunction to use
  Y           - Vector of outputs. For classification, use ±1 labels
  X           - Array of inputs
  k           - Sparsity parameter
  γ           - ℓ2-regularization parameter
  indices0    - (optional) Initial solution
  ΔT_max      - (optional) Maximum running time in seconds for the MIP solver. Default is 60s
  gap         - (optional) MIP solver accuracy

OUTPUT
  indices     - Indicates which features are used as regressors
  w           - Regression coefficients
  Δt          - Computational time (in seconds)
  status      - Solver status at termination
  Gap         - Optimality gap at termination
  cutCount    - Number of cuts needed in the cutting-plane algorithm
  """
function oa_formulation_bm(ℓ::LossFunction,
          Y::Union{Vector{Float64},SubArray{Float64}},
          X::Array{Float64,2}, k::Int, γ::Float64,
          solver::MathProgBase.AbstractMathProgSolver;
          indices0=find(x-> x<k/size(X,2), rand(size(X,2))),
          node_heuristics=false)

  if solver == UnsetSolver()
    error("You need to set a solver for the exact primal method. E.g.:
    `ExactPrimalCuttingPlane(0.1, 30.0, GurobiSolver)`")
  end

  n = size(Y, 1)
  p = size(X, 2)
  #Info array

  miop = Model(solver=solver)

  miop.ext[:heuristics_data] = # create storage

  s0 = zeros(p); s0[indices0]=1
  c0, ∇c0 = SubsetSelectionCIO.inner_op(ℓ, Y, X, s0, γ)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s)<=k)

  cutCount=1; bestObj=c0; bestSolution=s0[:];
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb)
    cutCount += 1
    c, ∇c = SubsetSelectionCIO.inner_op(ℓ, Y, X, getvalue(s), γ)
    if c<bestObj
      bestObj = c; bestSolution=getvalue(s)[:]
    end
    @lazyconstraint(cb, t>=c + dot(∇c, s-getvalue(s)))
  end
  addlazycallback(miop, outer_approximation)
  function myheuristic(cb)
    _, ∇c = SubsetSelectionCIO.inner_op(ℓ, Y, X, getvalue(s), γ)
    order = sortperm(∇c)
    promising_indices = collect(1:p)[order[1:k]]
    # hsolution = zeros(Int, p)
    # hsolution[promising_indices] = 1
    # setsolutionvalue(cb, s, hsolution)
    for i in promising_indices
      setsolutionvalue(cb, s[i], 1)
    end

    addsolution(cb)
  end
  if node_heuristics
    addheuristiccallback(miop, myheuristic)
  end

  status = solve(miop)
  Δt = getsolvetime(miop)

  if status != :Optimal
    Gap = 1 - JuMP.getobjbound(miop) /  getobjectivevalue(miop)
  else
    Gap = 0.0
  end

  if status == :Optimal
    bestSolution = getvalue(s)[:]
  end
  # Find selected regressors and run a standard linear regression with Tikhonov
  # regularization
  indices = find(s->s>0.5, bestSolution)
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  return indices, w, Δt, status, Gap, cutCount
end
