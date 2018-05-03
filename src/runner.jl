using RegressionBenchmarks, Distributions, MLDataUtils, DataFrames

srand(1)

# The scale of data we care about
nrange = collect(100:100:100)

# Number of features
d = 10
# What distribution will X come from
Xdist = MvNormal
# What form will the correlation in X take
Xcorr = NoCorrelation()
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
m = PrimalWithHeuristics()
# Get results
results_table = benchmark(bd, m)
# Save
resdir = joinpath(Pkg.dir("RegressionBenchmarks"), "results")
!isdir(resdir) && mkdir(resdir)
tstamp = round(Int64, time() * 1000)
datadir = joinpath(resdir, data2str(bd), method2str(m), "$(tstamp)")
!ispath(datadir) && mkpath(datadir)
df = convert(DataFrame, results_table)
colnames = [:accuracy, :false_detection, :train_r2, :test_r2, :time, :gamma]
names!(df, colnames)
insert!(df, 1, nrange, :nrange)
writetable(joinpath(datadir, "sparsity$(sparsity).csv"), df)
# Make plots
for field in ["accuracy", "false_detection", "test_r2", "time"]
    makeplot(datadir, field, df, sparsity)
end