include("exactprimal.jl")
include("subgradient.jl")

abstract type RegressionMethod end

"""
    ExactPrimalCuttingPlane

Use MIP solver to solve for optimal s.
"""
mutable struct ExactPrimalCuttingPlane <: RegressionMethod
    gamma::Float64
    time_limit::Float64
end
ExactPrimalCuttingPlane() = ExactPrimalCuttingPlane(0.0, 30.0)

"""
    PrimalWithHeuristics

Use MIP solver and supply it with node heuristics.
"""
mutable struct PrimalWithHeuristics <: RegressionMethod
    gamma::Float64
    time_limit::Float64
end
PrimalWithHeuristics() = PrimalWithHeuristics(0.0, 30.0)

"""
    RelaxDualSubgradient

Use subgradient descent on the dual of the convex relaxation.
"""
mutable struct RelaxDualSubgradient <: RegressionMethod
    gamma::Float64
    sr::SteppingRule
    maxiter::Int
end
RelaxDualSubgradient() = RelaxDualSubgradient(0.0, ConstantStepping(1e-3), 100)


function solve_problem(m::ExactPrimalCuttingPlane, X::Array{Float64,2}, Y, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma)
    indices0, w0
end

function solve_problem(m::PrimalWithHeuristics, X::Array{Float64,2}, Y, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma, node_heuristics=true)
    indices0, w0
end

function solve_problem(m::RelaxDualSubgradient, X::Array{Float64,2}, Y, k::Int)
  sparsity = SubsetSelection.Constraint(k)
  sp = subsetSelection_bm(SubsetSelection.OLS(), sparsity, Y, X, γ = m.gamma, sr = m.sr, maxIter = m.maxiter)
  sp.indices, sp.w
end
