using DelimitedFiles, DataFrames, Plots

fl_batch = readdlm("GMM/fl_batch.txt")
fl_svi = readdlm("GMM/fl_svi.txt")

time_batch, F_batch = fl_batch[1,:], fl_batch[2,:]
time_svi, F_svi = fl_svi[1,:], fl_svi[2,:]

gr()
function twiny(sp::Plots.Subplot)
    sp[:top_margin] = max(sp[:top_margin], 30Plots.px)
    plot!(sp.plt, inset = (sp[:subplot_index], bbox(0,0,1,1)))
    twinsp = sp.plt.subplots[end]
    twinsp[:xaxis][:mirror] = true
    twinsp[:background_color_inside] = RGBA{Float64}(0,0,0,0)
    Plots.link_axes!(sp[:yaxis], twinsp[:yaxis])
    twinsp
end
twiny(plt::Plots.Plot = current()) = twiny(plt[1])

plot(collect(2:length(F_svi)), F_svi[2:end],lw=3,color=:red, alpha=0.6, label="Minibatch SVI", legend=:topright, box = :on, grid = :off, xlabel="SVI iterations", left_margin = 10Plots.mm, right_margin = 5Plots.mm, top_margin = 10Plots.mm, bottom_margin = 5Plots.mm)
p = twiny()
plot!(p,collect(1:length(F_batch)),F_batch,lw=3,color=:green, alpha=0.6, label="Batch VMP", legend=:topleft, box = :on, grid = :off, ylabel="Free Energy", xlabel="VMP iterations")   
savefig("GMM/gmm_fe_compare.pdf")