"""
    accuracy(pred::Vector{Int}, truth::Vector{Int})

Returns proportion of indices in `truth` that are also in `pred`.
"""
function accuracy(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for t in truth
        (t in pred) && (detected += 1)
    end
    detected / length(truth)
end

"""
    falsepositive(pred::Vector{Int}, truth::Vector{Int})

Returns the proportion of indices in `pred` that are not in `truth`.
"""
function falsepositive(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for p in pred
        (p in truth) || (detected += 1)
    end
    detected / length(pred)
end

function mse(pred::YVector, truth::YVector)
    sum(abs2.(pred - truth))
end
function mse(pred::YVector, pt::Float64)
    sum(abs2.(pred - pt))
end
"""
    oosRsquared(pred::Vector{Float64}, truth::Vector{Float64}, train::Vector{Float64})

Computes the out-of-sample R^2.
"""
function oosRsquared(pred::Vector{Float64}, truth::YVector, train::YVector)
    1 - mse(pred, truth) / mse(pred, mean(train))
end

"""
    isRsquared(pred::Vector{Float64}, truth::Vector{Float64})

Computes the in-sample R^2.
"""
function isRsquared(pred::Vector{Float64}, truth::YVector)
    1 - mse(pred, truth) / mse(truth, mean(truth))
end

function predict_sparse(X::Array{Float64,2}, indices::Vector{Int}, w::Vector{Float64})
    X[:, indices] * w
end

function dist2str(::Union{Normal, MvNormal})
    "normal"
end
function dist2str(::Uniform)
    "uniform"
end
function dist2str(::BinChoice)
    "binchoice"
end
function dist2str(::NoNoise)
    "nonoise"
end
function corr2str(::NoCorrelation)
    "none"
end
function corr2str(c::MatrixCorrelation)
    "rho_$(c.coeff)"
end
function data2str(bd::BenchmarkData)
    "x_" * dist2str(bd.Xdata.dist) *
    "_corr_" * corr2str(bd.Xdata.corr) *
    "_w_" * dist2str(bd.wdist) *
    "_noise_" * dist2str(bd.noisedist) *
    "_snr_$(bd.SNR)" *
    "_d_$(bd.nfeatures)" *
    "_k_$(bd.sparsity)"
end
function Base.mkdir(bd::BenchmarkData)
    mkdir(data2str(bd))
end
function method2str(m::RegressionMethod)
    error("You need to define `method2str` for $m.")
end
function solver2str(s::CplexSolver)
    "cplex"
end
function solver2str(s::GurobiSolver)
    "gurobi"
end
function method2str(m::ExactPrimalCuttingPlane)
    "exact_primal_tlimit_$(m.time_limit)_$(solver2str(m.solver))"
end
function method2str(m::PrimalWithHeuristics)
    "node_heuristics_primal_tlimit_$(m.time_limit)_$(solver2str(m.solver))"
end
function stepping2str(sr::ConstantStepping)
  "conststep_$(sr.stepsize)"
end
function stepping2str(sr::PolyakStepping)
  "polyak_$(sr.initial_factor)"
end
function method2str(m::RelaxDualSubgradient)
    "relax_dual_subgradient_" *
    stepping2str(m.sr) *
    "_maxiter_$(m.maxiter)"
end
function method2str(m::RelaxDualCutting)
    "relax_dual_cut_" * "$(solver2str(m.solver))"
end
