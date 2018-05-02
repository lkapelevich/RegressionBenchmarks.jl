function makeplot(datadir::String, field::String, df::DataFrame, sparsity::Int)
    if field != "time"
        p = plot(df, x = "nrange", y = field,
          Scale.y_continuous(minvalue=0.0, maxvalue=1.0))
    else
      p = plot(df, x = "nrange", y = field)
    end
    draw(PNG(joinpath(datadir,
                    "$(field)_sparsity$(sparsity).png"
            ), 3inch, 3inch), p)
    nothing
end
