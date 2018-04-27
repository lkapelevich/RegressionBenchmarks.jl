abstract type RegressionMethod end

type ExactPrimalCuttingPlane <: RegressionMethod
    gamma::Float64
    time_limit::Float64
end
ExactPrimalCuttingPlane() = ExactPrimalCuttingPlane(0.0, 30.0)
struct RelaxPrimalCuttingPane <: RegressionMethod end
struct RelaxDualSubgradient <: RegressionMethod end
struct PrimalWithHeuristics <: RegressionMethod end


function solve_problem(m::ExactPrimalCuttingPlane, X::Array{Float64,2}, Y, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(), Y, X, sparsity, 1 / m.gamma)
    indices0, w0
end
