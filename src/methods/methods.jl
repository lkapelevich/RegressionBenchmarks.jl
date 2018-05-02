abstract type RegressionMethod end

mutable struct ExactPrimalCuttingPlane <: RegressionMethod
    gamma::Float64
    time_limit::Float64
end
ExactPrimalCuttingPlane() = ExactPrimalCuttingPlane(0.0, 30.0)
mutable struct PrimalWithHeuristics <: RegressionMethod
    gamma::Float64
    time_limit::Float64
end
PrimalWithHeuristics() = PrimalWithHeuristics(0.0, 30.0)
struct RelaxPrimalCuttingPane <: RegressionMethod end
struct RelaxDualSubgradient <: RegressionMethod end

include("exactprimal.jl")

function solve_problem(m::ExactPrimalCuttingPlane, X::Array{Float64,2}, Y, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(), Y, X, sparsity, 1 / m.gamma)
    indices0, w0
end

function solve_problem(m::PrimalWithHeuristics, X::Array{Float64,2}, Y, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, 1 / m.gamma, node_heuristics=true)
    indices0, w0
end
