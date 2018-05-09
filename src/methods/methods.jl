include("exactprimal.jl")
include("subgradient.jl")
include("cuttingdual.jl")

abstract type RegressionMethod end
abstract type WarmStartMethod <: RegressionMethod end
struct NoWarmStart <: WarmStartMethod end

"""
    ExactPrimalCuttingPlane

Use MIP solver to solve for optimal s.
"""
mutable struct ExactPrimalCuttingPlane <: RegressionMethod
    gamma::Float64
    time_limit::Float64
    solver::MathProgBase.AbstractMathProgSolver
    warm_start::WarmStartMethod
end
ExactPrimalCuttingPlane() = ExactPrimalCuttingPlane(0.0, 30.0, UnsetSolver, NoWarmStart())
"""
    ExactPrimalCuttingPlane(gamma::Float64, time_limit::Float64,
            stype::Type{S}, warm_start::WarmStartMethod) where {S <: MathProgBase.AbstractMathProgSolver}
"""
function ExactPrimalCuttingPlane(gamma::Float64, time_limit::Float64, stype::Type{S}, warm_start::WarmStartMethod) where {S <: MathProgBase.AbstractMathProgSolver}
      solver = getsolver(stype, time_limit)
      ExactPrimalCuttingPlane(gamma, time_limit, solver, warm_start)
end
function ExactPrimalCuttingPlane(stype::Type{S}) where {S <: MathProgBase.AbstractMathProgSolver}
      ExactPrimalCuttingPlane(0.0, 30.0, stype, NoWarmStart())
end

"""
    PrimalWithHeuristics

Use MIP solver and supply it with node heuristics.
"""
mutable struct PrimalWithHeuristics <: RegressionMethod
    gamma::Float64
    time_limit::Float64
    solver::MathProgBase.AbstractMathProgSolver
    warm_start::WarmStartMethod
end
PrimalWithHeuristics() = PrimalWithHeuristics(0.0, 30.0, UnsetSolver, NoWarmStart())
"""
    PrimalWithHeuristics(gamma::Float64, time_limit::Float64,
            stype::Type{S}, warm_start::WarmStartMethod) where {S <: MathProgBase.AbstractMathProgSolver}
"""
function PrimalWithHeuristics(gamma::Float64, time_limit::Float64, stype::Type{S}, warm_start::WarmStartMethod) where {S <: MathProgBase.AbstractMathProgSolver}
      solver = getsolver(stype, time_limit)
      ExactPrimalCuttingPlane(gamma, time_limit, solver, warm_start)
end
function PrimalWithHeuristics(stype::Type{S}) where {S <: MathProgBase.AbstractMathProgSolver}
      ExactPrimalCuttingPlane(0.0, 30.0, stype,  NoWarmStart())
end

"""
    RelaxDualSubgradient

Use subgradient descent on the dual of the convex relaxation.
"""
mutable struct RelaxDualSubgradient{SR <: SteppingRule} <: WarmStartMethod
    gamma::Float64
    sr::SR
    maxiter::Int
end
RelaxDualSubgradient() = RelaxDualSubgradient(0.0, ConstantStepping(1e-3), 100)

"""
    RelaxDualCutting

Use Kelley's cuts on dual of convex relaxation.
"""
mutable struct RelaxDualCutting <: WarmStartMethod
    gamma::Float64
    solver::MathProgBase.AbstractMathProgSolver
end
RelaxDualCutting() = RelaxDualCutting(0.0, UnsetSolver())

function solve_problem(::RegressionMethod, ::Array{Float64,2}, ::YVector, ::Int)
    error("You need to define `solve_problem` for $m.")
end

function solve_problem(m::ExactPrimalCuttingPlane, X::Array{Float64,2}, Y::YVector, sparsity::Int)
    if m.warm_start != NoWarmStart()
        inds, _ = solve_problem(m.warm_start, X, Y, sparsity)
        indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma, m.solver, indices0 = inds)
    else
        indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma, m.solver)
    end
    indices0, w0
end

function solve_problem(m::PrimalWithHeuristics, X::Array{Float64,2}, Y::YVector, sparsity::Int)
    if m.warm_start != NoWarmStart()
        inds, _ = solve_problem(m.warm_start, X, Y, sparsity)
        indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma, m.solver, node_heuristics=true, indices0=inds)
    else
        indices0, w0, Δt, status, Gap, cutCount = oa_formulation_bm(SubsetSelection.OLS(), Y, X, sparsity, m.gamma, m.solver, node_heuristics=true)
    end
    indices0, w0
end

function solve_problem(m::RelaxDualSubgradient, X::Array{Float64,2}, Y::YVector, k::Int)
    sparsity = SubsetSelection.Constraint(k)
    sp = subsetSelection_bm(SubsetSelection.OLS(), sparsity, Y, X, γ = m.gamma, sr = m.sr, maxIter = m.maxiter)
    sp.indices, sp.w
end

function solve_problem(m::RelaxDualCutting, X::Array{Float64,2}, Y::YVector, k::Int)
    solve_dualcutting(X, Y, k, m.gamma, m.solver)
end
