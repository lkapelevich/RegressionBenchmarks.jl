abstract type RegressionMethod end

struct ExactPrimalCuttingPlane <: RegressionMethod
    gamma::Float64
end
struct RelaxPrimalCuttingPane <: RegressionMethod end
struct RelaxDualSubgradient <: RegressionMethod end


function solve_problem(m::ExactPrimalCuttingPlane, data::BenchmarkData, sparsity::Int)
    indices0, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(), data.Y, data.X, sparsity, m.gamma)
end
