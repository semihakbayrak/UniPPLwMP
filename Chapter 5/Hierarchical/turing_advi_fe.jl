using Distributions, Turing, AdvancedVI, Random
using DelimitedFiles, DataFrames

Random.seed!(123);

# -------------------------------------------
# Dataset
# -------------------------------------------

dataset = [[58.7226956168386, 45.13976354300799, 34.891234306106206, 22.049813810571656, 18.029311822124683, 42.714517401378004, 26.86775400403366, 32.107230568182345],
 [4.606339701921808, -0.43877921447069923, -0.8893574689730634],
 [6.478451882763512, 2.888032369033442, -7.49813190569346, -14.758169441040984, -14.478560265189357, -15.401117821926048, -18.672627310307654],
 [1.846619359098037, -2.68986972978257, 10.054981057055565, 23.763077651153214, -12.555004304724793],
 [6.040319868293942, 19.822216736962826, -8.83951499143022, 2.1115471241811163, -2.4026795198407163, 19.997114297241556, -9.985174142931942, -0.3973994620971628],
 [-8.68576178873407, -1.791407472467292, 9.954292431463115, -11.37327663982918, -2.78835504435457, -1.6674794730107032, 17.851944050272632, 5.547874888293355, -1.5037350724506187],
 [33.31815939613223, -13.432641838306846, 3.5993734745928982],
 [8.268877742477276, 35.26754519612162, 33.05402194461256]]

# -------------------------------------------
# Turing Model Specification
# -------------------------------------------
@model function eight_school(y)
    α ~ Gamma(.1,1/.1)
    β ~ Gamma(.1,1/.1)
    μ ~ Normal(0.,sqrt(10))
    s ~ Gamma(.1,1/1.)
    x = Vector{Any}(undef, 8)
    w = Vector{Any}(undef, 8)
    for i=1:8
        x[i] ~ Normal(μ,sqrt(1/s))
        w[i] ~ Gamma(α,1/β)
        for j=1:length(y[i])
            y[i][j] ~ Normal(x[i],sqrt(1/w[i]))
        end
    end
end

# Instantiate model
sc_model = eight_school(dataset)

# -------------------------------------------
# ADVI inference
# -------------------------------------------
# Mean-field, 1 sample per iteration to estimate gradient of ELBO, max iteration number = 5000
advi = ADVI(1, 5000)
q = vi(sc_model, advi)
FE_estimate = -AdvancedVI.elbo(advi, q, sc_model, 1000)

# run_time = 0.
# run_time += @elapsed advi = ADVI(1, 100)
# run_time += @elapsed q = vi(sc_model, advi)
# FE_estimate_ADVI = [-AdvancedVI.elbo(advi, q, sc_model, 1000)]
# run_time_ADVI = [run_time]
# for _=1:100
#     run_time += @elapsed q = vi(sc_model, advi, q)
#     push!(FE_estimate_ADVI, -AdvancedVI.elbo(advi, q, sc_model, 1000))
#     push!(run_time_ADVI, run_time)
# end

# Save free energy and run time values
#writedlm("Hierarchical/advi_fe.txt", [run_time_ADVI, FE_estimate_ADVI])
writedlm("Hierarchical/advi_fe.txt", FE_estimate)
