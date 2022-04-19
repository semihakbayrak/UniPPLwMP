using Turing, Plots, Random, BenchmarkTools
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff);
Turing.setrdcache(true);

Random.seed!(0);
#Generate data
T = 120

w1, w2, w3 = 0.1, 0.25, 1

x_data = [randn()]
y_data = [x_data[end]+0.1*randn()]
for t=2:25
    append!(x_data, x_data[end] + sqrt(1/w1)*randn())
    append!(y_data, x_data[end] + randn())
end
for t=26:75
    append!(x_data, x_data[end] + sqrt(1/w2)*randn())
    append!(y_data, x_data[end] + randn())
end
for t=76:T
    append!(x_data, x_data[end] + sqrt(1/w3)*randn())
    append!(y_data, x_data[end] + randn())
end

function run_hmc()
    @model function SSSM(y)
        vars = [10, 4, 1]
        T = length(y)
        z = tzeros(Int,T-1)
        x = Vector(undef, T)
        #M = Vector{Vector}(undef,3) # Transition matrix
        M_1 ~ Dirichlet([100,1,1])
        M_2 ~ Dirichlet([1,100,1])
        M_3 ~ Dirichlet([1,1,100])
        M = [M_1, M_2, M_3] # Transition matrix
        
        z[1] ~ Categorical(3)
        x[1] ~ Normal()
        y[1] ~ Normal(x[1],sqrt(1))
        for t = 2:T-1
            x[t] ~ Normal(x[t-1],sqrt(vars[z[t-1]]))
            y[t] ~ Normal(x[t],sqrt(1))
            z[t] ~ Categorical(vec(M[z[t-1]]))
        end
        x[T] ~ Normal(x[T-1],sqrt(vars[z[T-1]]))
        y[T] ~ Normal(x[T],sqrt(1))
    end

    gibbs = Gibbs(HMC(0.2,20,:x,:M_1,:M_2,:M_3),PG(50,:z))
    chain = sample(SSSM(y_data),gibbs,1000)
end

bench_hmc = @benchmarkable run_hmc()
run(bench_hmc, samples=10, seconds=15000)