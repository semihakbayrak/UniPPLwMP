using ForneyLab, LinearAlgebra, Random, Plots, Flux.Optimise
import ForneyLab: unsafeMean, unsafeVar, step!

Random.seed!(123)

dataset = [4, 
5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 
1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 
1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 
0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 
1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
1, 0, 0, 1, 0, 1]

T = length(dataset)

# -------------------------------------------
# Online Exact Inference (Treat the problem as if it was a stationary process)
# -------------------------------------------
# Model Specification
graph = FactorGraph()

@RV λ ~ Gamma(placeholder(:a), placeholder(:b))

@RV y ~ Poisson(λ)
placeholder(y, :y)

# Algorithm construction
algo = messagePassingAlgorithm(λ)
src_code = algorithmSourceCode(algo)
eval(Meta.parse(src_code))

# Run Inference
marginals = Dict(:λ => ProbabilityDistribution(Univariate, Gamma, a=1., b=1.))

meanvalsBP = []

for t=1:T   
    data = Dict(:y => dataset[t], :a => marginals[:λ].params[:a], :b => marginals[:λ].params[:b])

    step!(data, marginals)
    push!(meanvalsBP,unsafeMean(marginals[:λ]))
end

# -------------------------------------------
# Online SVI - Robins&Monro satisfied (Treat the problem as if it was a stationary process)
# -------------------------------------------
# Model Specification
graph = FactorGraph()

@RV λ ~ Gamma(1., 1.)

@RV λ_t ~ Svi(λ, opt=ForgetDelayDescent(0.,1.), q=ProbabilityDistribution(Univariate, Gamma, a=1., b=1.), batch_size=1, dataset_size=T)

@RV y ~ Poisson(λ_t)
placeholder(y, :y)
;

# Algorithm construction
PosteriorFactorization(λ_t)
algo = messagePassingAlgorithm(free_energy=false, id=:MF)
src_code = algorithmSourceCode(algo, free_energy=false)
eval(Meta.parse(src_code))

# Run Inference
marginals = Dict(:λ => ProbabilityDistribution(Univariate, Gamma, a=1., b=1.),
                 :λ_t => ProbabilityDistribution(Univariate, Gamma, a=1., b=1.))

meanvalsSVI1 = []

for t=1:T
    data = Dict(:y => dataset[t])

    stepMFposteriorfactor_1!(data, marginals)
    push!(meanvalsSVI1,unsafeMean(marginals[:λ]))
end

# -------------------------------------------
# Online SVI - Exponential Weighting
# -------------------------------------------
# Model Specification
graph = FactorGraph()

@RV λ ~ Gamma(1., 1.)

@RV λ_t ~ Svi(λ, opt=Descent(0.1), q=ProbabilityDistribution(Univariate, Gamma, a=1., b=1.), batch_size=1, dataset_size=T)

@RV y ~ Poisson(λ_t)
placeholder(y, :y)
;

# Algorithm construction
PosteriorFactorization(λ_t)
algo = messagePassingAlgorithm(free_energy=false, id=:MF)
src_code = algorithmSourceCode(algo, free_energy=false)
eval(Meta.parse(src_code))

# Run Inference
marginals = Dict(:λ => ProbabilityDistribution(Univariate, Gamma, a=1., b=1.),
                 :λ_t => ProbabilityDistribution(Univariate, Gamma, a=1., b=1.))

meanvalsSVI2 = []

for t=1:T
    data = Dict(:y => dataset[t])

    stepMFposteriorfactor_1!(data, marginals)
    push!(meanvalsSVI2,unsafeMean(marginals[:λ]))
end

scatter(collect(1:T),dataset,color=:black,size=(900,200),legend=:topright, 
        xtick=([1, 15, 29, 43, 57, 71, 85, 99, 112], [1851, 1865, 1879, 1893, 1907, 1921, 1935, 1949, 1962]), 
        xlabel="Years", ylabel="Accidents", label="Observations", left_margin = 10Plots.mm, bottom_margin = 5Plots.mm)
plot!(meanvalsBP,lw=2, color=:blue, alpha=0.5, label="Online BP", size=(900,400))
plot!(meanvalsSVI1, lw=2, color=:green, alpha=0.5, label="Online SVI ρₜ = 1/t")
plot!(meanvalsSVI2, lw=2, color=:red, alpha=0.5, label="Online SVI ρₜ = 0.1")
savefig("NonstationaryProcess/Nonstationary.pdf")