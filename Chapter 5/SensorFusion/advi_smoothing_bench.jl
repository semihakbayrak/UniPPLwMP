using Turing, Random, LinearAlgebra, Plots, Flux.Optimise, DelimitedFiles, DataFrames, Bijectors
using Turing: Variational
using Bijectors: Scale, Shift
using AdvancedVI
using ComponentArrays, UnPack
using ReverseDiff, Memoization
using BenchmarkTools

Random.seed!(1234);
Turing.setadbackend(:reversediff);
Turing.setrdcache(true);

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

function run_advi()
# Turing Model
@model function model(y)
    W, R = diagm(0=>ones(4)), diagm(0=>ones(3))
    z = Vector{Vector}(undef, T)
    
    z[1] ~ MvNormal(zeros(4), diagm(0=>ones(4)))
    y[1] ~ MvNormal(f(z[1]), R)
    
    for t=2:T
        z[t] ~ MvNormal(A*z[t-1],W)
        y[t] ~ MvNormal(f(z[t]),R)
    end
end;

m = model(observation_list);

# -------------------------------------------
# Inference
# -------------------------------------------
d = T*4
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d));

to_constrained = inv(bijector(m));

proto_arr = ComponentArray(
    L = zeros(d, d),
    b = zeros(d)
)
proto_axes = proto_arr |> getaxes
num_params = length(proto_arr)

function getq(θ)
    L, b = begin
        @unpack L, b = ComponentArray(θ, proto_axes)
        LowerTriangular(L), b
    end
    # For this to represent a covariance matrix we need to ensure that the diagonal is positive.
    # We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
    D = Diagonal(diag(L))
    A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries
    
    b = to_constrained ∘ Shift(b; dim = Val(1)) ∘ Scale(A; dim = Val(1))
    
    return transformed(base_dist, b)
end

advi = ADVI(1, 6000)

q_full_normal = vi(m, advi, getq, randn(num_params));

end

bench_advi = @benchmarkable run_advi()
benchmark_result = run(bench_advi, samples=100, seconds=10000)