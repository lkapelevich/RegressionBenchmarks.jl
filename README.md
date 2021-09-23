# RegressionBenchmarks

(This was a class project and not meant to be maintained or used for anything super serious)

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
                  n = collect(100:20:500),
                  nfeatures = 1000,
                  sparsity = 10)
# Model
m = ExactPrimalCuttingPlane()
# Results
results_table = benchmark(bd, m)
```

## References
Bertsimas, Dimitris, and Bart Van Parys. "Sparse high-dimensional regression: Exact scalable algorithms and phase transitions." _arXiv preprint arXiv:1709.10029 (2017)._

Bertsimas, Dimitris, Jean Pauphilet, and Bart Van Parys. "Sparse Classification and Phase Transitions: A Discrete Optimization Perspective." _arXiv preprint arXiv:1710.01352 (2017)._
