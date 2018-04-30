using RegressionBenchmarks, Distributions, MLDataUtils, DataFrames

srand(1)

# The scale of data we care about
nrange = collect(100:100:1000)

# Number of features
d = 10
# What distribution will X come from
Xdist = MvNormal
# What form will the correlation in X take
Xcorr = MatrixCorrelation(0.1)
# Create an object for our X data
Xdata = XData(Xdist, Xcorr, d)
# True sparsity
sparsity = 5

# Create our data object
bd = BenchmarkData(Xdata = Xdata,
                  wdist = BinChoice(),
                  noisedist = NoNoise(),
                  SNR = 0.0,
                  n = nrange,
                  nfeatures = d,
                  sparsity = sparsity)
# Choose model
m = ExactPrimalCuttingPlane()
# Get results
results_table = benchmark(bd, m)
# Save
resdir = joinpath(Pkg.dir("RegressionBenchmarks"), "results")
!isdir(resdir) && mkdir(resdir)
datadir = joinpath(resdir, RegressionBenchmarks.data2str(bd))
!isdir(datadir) && mkdir(datadir)
df = convert(DataFrame, results_table)
colnames = [:accuracy, :false_detection, :train_r2, :test_r2, :time, :gamma]
names!(df, colnames)
insert!(df, 1, nrange, :nrange)
writetable(joinpath(datadir, "sparsity$(sparsity).csv"), df)
# Make plots
for field in ["accuracy", "false_detection", "test_r2", "time"]
    makeplot(datadir, field, df, sparsity)
end
