using DelimitedFiles, DataFrames, Plots

fe_advi = readdlm("Hierarchical/advi_fe.txt")
fe_cvi = readdlm("Hierarchical/cvi_fe.txt")

gr()
time_cvi, F_cvi = fe_cvi[1,:], fe_cvi[2,:]
plot(collect(2:length(F_cvi)), F_cvi[2:end],lw=3,color=:red, alpha=0.6, label="ForneyLab", legend=:topright, box = :on, grid = :off, xlabel="ForneyLab VMP iterations", ylabel="Free Energy", left_margin = 10Plots.mm, bottom_margin = 5Plots.mm)
plot!(collect(2:length(F_cvi)), fe_advi .* ones(length(F_cvi)-1), color=:black, alpha=1., lw=2, linestyle=:dash, label="Turing ADVI")
savefig("Hierarchical/hierarchical_fe_compare.pdf")
