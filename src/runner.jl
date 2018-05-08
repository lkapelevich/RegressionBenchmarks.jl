using RegressionBenchmarks, Distributions, MLDataUtils, DataFrames

srand(1)

# The scale of data we care about
nrange = collect(100:20:500)

# Number of features
d = 1000

# What form will the correlation in X take
Xcorr = NoCorrelation()
# True sparsity
sparsity = 10

noisedist = Normal(0.0, 1.0)

# What distribution will X come from
for Xdist in [Uniform, MvNormal]
    for wdist in [BinChoice(), Uniform(), Normal(0.0, 1.0)]
        for SNR in [20.0, 5.0, 1.0, 0.5]
              for d in [1000, 5000, 10000]

                  # Create an object for our X data
                  Xdata = XData(Xdist, Xcorr, d)

                  # Create our data object
                  bd = BenchmarkData(Xdata = Xdata,
                                    wdist = wdist,
                                    noisedist = noisedist,
                                    SNR = SNR,
                                    n = nrange,
                                    nfeatures = d,
                                    sparsity = sparsity)
                  # Choose model
                  for m in [ExactPrimalCuttingPlane(0.0, 300.0)]
                      # Get results
                      results_table, val_results = benchmark(bd, m)
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
                      writetable(joinpath(datadir, "sparsity$(sparsity)_results.csv"), df)
                      # Archive validation results also
                      open(joinpath(datadir, "sparsity$(sparsity)_validation.csv"), "w") do io
                          RegressionBenchmarks.valid2io(io, val_results, nrange)
                      end
                      # Make plots
                      for field in ["accuracy", "false_detection", "test_r2", "time"]
                          makeplot(datadir, field, df, sparsity)
                      end
                  end # method
              end # d
        end  # SNR
    end  # wdist
end # xdist
