# RegressionBenchmarks

This package was created to make benchmarking different approaches to the best subset problem easier, and archive results.

We separate the characteristics that our data has from the method we use to solve the regression problem.

## Data

We'll say that input data can be fully described by the following features
* What is the number of features?
* What distribution does X come from?
* How are featuers in X correlated?
* What values can our regression coefficients take?
* What is the noise distribution in our data?
* How small is the signal to noise ratio?

## Methods
* Exact MIP (primal)
* Subgradient ascent in dual space
* Cutting planes in dual space
* Exact MIP (primal) with user node heuristics

## Running
Got a new method or new set of data of interest? Define them and run:

```julia
# Data
bd = BenchmarkData(Xdata = Xdata(MvNormal, NoCorrelation()),
                  wdist = BinChoice(),
                  noisedist = NoNoise(),
                  SNR = 0.0,
                  n = nrange,
                  nfeatures = d,
                  sparsity = sparsity)
# Model
m = ExactPrimalCuttingPlane()
# Results
results_table = benchmark(bd, m)
```
