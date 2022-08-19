using ForneyLab, Random, LinearAlgebra, Plots, Flux.Optimise, DelimitedFiles, DataFrames
using ForneyLab: unsafeMean, unsafeCov
import ForneyLab.step!
using BenchmarkTools

Random.seed!(1234);

# -------------------------------------------
# Dataset
# -------------------------------------------
sensors = readdlm("SensorFusion/sensors.txt")
sensor1, sensor2, sensor3 = sensors[1,:], sensors[2,:], sensors[3,:]
observation = readdlm("SensorFusion/observation.txt")
position = readdlm("SensorFusion/position.txt")
T = 15
observation_list = []
for t=1:T
    push!(observation_list, observation[t,:])
end

# -------------------------------------------
# Model Specification
# -------------------------------------------
# Newtonian dynamics assumption
A = [1. 0. 1. 0.; 0. 1. 0. 1.; 0. 0. 1. 0.; 0. 0. 0. 1.]
B = [1. 0. 0. 0.; 0. 1. 0. 0.]

function f(z)       
    pos = B*z
    o1 = sqrt(sum((pos-sensor1).^2))
    o2 = sqrt(sum((pos-sensor2).^2))
    o3 = sqrt(sum((pos-sensor3).^2))
    o = [o1,o2,o3]
end

function run_cvi()
# Factor graph
graph = FactorGraph()

W = diagm(0=>ones(4))
R = diagm(0=>ones(3))

z = Vector{Variable}(undef, T)
x = Vector{Variable}(undef, T)
y = Vector{Variable}(undef, T)

@RV z[1] ~ GaussianMeanVariance(zeros(4), diagm(0=>ones(4)))
@RV x[1] ~ Cvi(z[1],g=f,opt=Descent(0.1),num_samples=1000,num_iterations=100)
@RV y[1] ~ GaussianMeanPrecision(x[1],R)
placeholder(y[1], :y, dims=(3,), index=1)

for t=2:T
    @RV z[t] ~ GaussianMeanPrecision(A*z[t-1],W)
    @RV x[t] ~ Cvi(z[t],g=f,opt=Descent(0.1),num_samples=1000,num_iterations=100)
    @RV y[t] ~ GaussianMeanPrecision(x[t],R)
    placeholder(y[t], :y, dims=(3,), index=t)
end

# -------------------------------------------
# Inference
# -------------------------------------------
# Algorithm construction
# Specify structured factorizations in recognition distribution
pfz = PosteriorFactorization()
q_z = PosteriorFactor(z, id=:z)
algo_struct = messagePassingAlgorithm(id=:Struct, free_energy=false)
# Generate source code
code_struct = algorithmSourceCode(algo_struct, free_energy=false);
# Load algorithm
eval(Meta.parse(code_struct));

# Messages needs to be initiated for CVI
messages = Vector{Message}(undef, 194)
for i=1:194
    messages[i] = Message(Multivariate, GaussianMeanPrecision, m=zeros(4), w=0.01*diagm(0=>ones(4)))
end

n_its = 5
data = Dict(:y => observation_list)
marginals = Dict()

marginals[:R] = diagm(0=>ones(3))
marginals[:W] = diagm(0=>ones(4))

# Run algorithm
for i = 1:n_its
    stepStructz!(data, marginals, messages)
end

end

bench_cvi = @benchmarkable run_cvi()
benchmark_result = run(bench_cvi, samples=100, seconds=10000)