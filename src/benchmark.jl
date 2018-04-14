function benchmark(data, method)
    best_params = validate!(data, method)
    save_params(best_params)
    solutions = benchmark(data, method, best_params)
end



srand(1)
n = 20
d = 10
sparsity = 5
μ = 0.0
Σ = 0.1 * ones(d, d)
@inbounds for i = 1:d
    Σ[i, i] = 1.0
end

bd = getdata(Xdist = MvNormal(μ * ones(d), Σ), wdist = BinChoice(), n = n, p = d, k = sparsity)
