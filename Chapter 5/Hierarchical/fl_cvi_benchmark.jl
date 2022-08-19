using ForneyLab, Flux.Optimise, Random
using DelimitedFiles, DataFrames
using BenchmarkTools

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

global N = 0
for i=1:8 global N += length(dataset[i]) end

# -------------------------------------------
# ForneyLab Model Specification
# -------------------------------------------
function run_cvi()
graph = FactorGraph()

f(α) = α

@RV α ~ Gamma(.1,.1)
@RV α_ ~ Cvi(α,g=f,opt=ADAM(),num_samples=1000,num_iterations=10000)
@RV β ~ Gamma(.1,.1)
@RV μ ~ GaussianMeanVariance(0,10)
@RV s ~ Gamma(.1,1.)

x = Vector{Variable}(undef, 8)
w = Vector{Variable}(undef, 8)
y = Vector{Variable}(undef, N)

n_count = 0
for i=1:8
    @RV x[i] ~ GaussianMeanPrecision(μ,s)
    @RV w[i] ~ Gamma(α_,β)
    for n=1:length(dataset[i])
        n_count += 1
        @RV y[n_count] ~ GaussianMeanPrecision(x[i],w[i])
        placeholder(y[n_count], :y, index=n_count)
    end
end

# -------------------------------------------
# Approximate Distribution Factorization
# -------------------------------------------

pfz = PosteriorFactorization()

q_α = PosteriorFactor(α, id=:α)
q_β = PosteriorFactor(β, id=:β)
q_μ = PosteriorFactor(μ, id=:μ)
q_s = PosteriorFactor(s, id=:s)
q_x, q_w = Vector{PosteriorFactor}(undef, 8), Vector{PosteriorFactor}(undef, 8)
for i=1:8
    q_x[i] = PosteriorFactor(x[i],id=:x_*i)
    q_w[i] = PosteriorFactor(w[i],id=:w_*i)
end

run_time = 0.
# Build the algorithm
algo = messagePassingAlgorithm(free_energy=false)

# Generate source code
source_code = algorithmSourceCode(algo, free_energy=false);
eval(Meta.parse(source_code));

# -------------------------------------------
# Execute Inference
# -------------------------------------------
# Prepare posterior factors
marginals = Dict(:α => vague(Gamma), :α_ => vague(Gamma), :β => vague(Gamma), :s => vague(Gamma),
                 :μ => vague(GaussianMeanVariance))

for i=1:8
    marginals[:w_*i] = vague(Gamma)
    marginals[:x_*i] = vague(GaussianMeanPrecision)
end

y_data = []
for i=1:8
    y_data = [y_data;dataset[i]]
end
data = Dict(:y => y_data)

n_its = 10
for j=1:n_its
    for i=1:8
        step!(:x_*i,data, marginals)
    end
    for i=1:8
        step!(:w_*i,data, marginals)
    end
    stepμ!(data, marginals)
    steps!(data, marginals)
    stepβ!(data, marginals)
    stepα!(data, marginals)
end

end

bench_cvi = @benchmarkable run_cvi()
run(bench_cvi, samples=100, seconds=10000)
