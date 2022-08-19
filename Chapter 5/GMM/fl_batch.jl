using ForneyLab, LinearAlgebra, Flux.Optimise, Plots, Random, MultivariateStats
using Distributions: MvNormal, MixtureModel, pdf
using ForneyLab: unsafeMean, unsafeCov
using DelimitedFiles, DataFrames

Random.seed!(0);
global run_time = 0.

# -------------------------------------------
# Preprocessing Iris Data
# -------------------------------------------
N = 150 # number of data points

DATA, DATA_y = zeros(150,4), zeros(150)
text = readlines("GMM/iris-dataset.txt")

for i=1:length(text) 
    line = text[i]
    DATA[i,:] = parse.(Float64, (split.(line, ",")[1:end-1]))
    if split.(line, ",")[end] == "Iris-setosa"
        DATA_y[i] = 1
    elseif split.(line, ",")[end] == "Iris-versicolor"
        DATA_y[i] = 2
    else
        DATA_y[i] = 3
    end
end

P = fit(PCA,DATA';maxoutdim=2)
dataset = MultivariateStats.transform(P,DATA')

data_all = []
for n=1:N
    push!(data_all,[dataset[:,n];DATA_y[n]])
end

shuffle!(data_all)
data_x, data_y = [], []
for n=1:N
    push!(data_x,data_all[n][1:2])
    push!(data_y,data_all[n][3])
end

# -------------------------------------------
# ForneyLab Batch Model Specification
# -------------------------------------------
graph = FactorGraph()

@RV _pi ~ Dirichlet([50.,50.,50.])
@RV m_1 ~ GaussianMeanVariance(zeros(2), diagm(0=>ones(2)))
@RV W_1 ~ Wishart(diagm(0=>ones(2)),2)
@RV m_2 ~ GaussianMeanVariance(zeros(2), diagm(0=>ones(2)))
@RV W_2 ~ Wishart(diagm(0=>ones(2)),2)
@RV m_3 ~ GaussianMeanVariance(zeros(2), diagm(0=>ones(2)))
@RV W_3 ~ Wishart(diagm(0=>ones(2)),2)

z = Vector{Variable}(undef, N)
y = Vector{Variable}(undef, N)
for i = 1:N
    @RV z[i] ~ Categorical(_pi)
    @RV y[i] ~ GaussianMixture(z[i], m_1, W_1, m_2, W_2, m_3, W_3)
    
    placeholder(y[i], :y, index=i, dims=(2,))
end

# -------------------------------------------
# Approximate Distribution Factorization
# -------------------------------------------
pfz = PosteriorFactorization()

q_m1 = PosteriorFactor(m_1, id=:M1MF)
q_W1 = PosteriorFactor(W_1, id=:W1MF)
q_m2 = PosteriorFactor(m_2, id=:M2MF)
q_W2 = PosteriorFactor(W_2, id=:W2MF)
q_m3 = PosteriorFactor(m_3, id=:M3MF)
q_W3 = PosteriorFactor(W_3, id=:W3MF)
q_pi = PosteriorFactor(_pi, id=:PIMF)
q_z = Vector{PosteriorFactor}(undef, N)
for t=1:N
    q_z[t] = PosteriorFactor(z[t],id=:Z_*t)
end

# Build the algorithm
global run_time += @elapsed algo = messagePassingAlgorithm(free_energy=true)

# Generate source code
global run_time += @elapsed source_code = algorithmSourceCode(algo, free_energy=true)
eval(Meta.parse(source_code))

# -------------------------------------------
# Execute Inference
# -------------------------------------------
# Prepare posterior factors
marginals = Dict(:_pi => vague(Dirichlet,3),
                 :m_1 => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[1.,0.], v=10.0*diagm(0=>ones(2))),
                 :m_2 => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[-1.,0.], v=10.0*diagm(0=>ones(2))),
                 :m_3 => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(2), v=10.0*diagm(0=>ones(2))),
                 :W_1 => ProbabilityDistribution(MatrixVariate, Wishart, v=diagm(0=>1000*ones(2)), nu=2),
                 :W_2 => ProbabilityDistribution(MatrixVariate, Wishart, v=diagm(0=>1000*ones(2)), nu=2),
                 :W_3 => ProbabilityDistribution(MatrixVariate, Wishart, v=diagm(0=>1000*ones(2)), nu=2))

# Execute algorithm
n_its = 10
F_batch, time_batch = Float64[], Float64[]

data = Dict(:y => data_x)

for i = 1:n_its
    global run_time += @elapsed for j=1:N
        step!(:Z_*j,data, marginals)
    end
    global run_time += @elapsed stepPIMF!(data, marginals)
    global run_time += @elapsed stepM1MF!(data, marginals)
    global run_time += @elapsed stepW1MF!(data, marginals)
    global run_time += @elapsed stepM2MF!(data, marginals)
    global run_time += @elapsed stepW2MF!(data, marginals)
    global run_time += @elapsed stepM3MF!(data, marginals)
    global run_time += @elapsed stepW3MF!(data, marginals)
        
    # Store variational free energy
    global run_time += @elapsed F = freeEnergy(data, marginals)
    push!(F_batch, F)
    push!(time_batch, run_time)
end

# Save free energy and run time values
writedlm("GMM/fl_batch.txt", [time_batch, F_batch]) 

# # -------------------------------------------
# # Plot
# # -------------------------------------------
# color_list = [:blue, :yellow, :red]
# scatter((data_x[1][1],data_x[1][2]),color=color_list[argmax(marginals[:z_1].params[:p])], legend=false)
# for n=2:N
#     scatter!((data_x[n][1],data_x[n][2]),color=color_list[argmax(marginals[:z_*n].params[:p])])
# end
# X = range(-4, 4, length=200)
# Y = range(-2, 2, length=200)
# Z1 = [pdf(MvNormal(unsafeMean(marginals[:m_1]),Matrix(Hermitian(inv(unsafeMean(marginals[:W_1]))))), [x,y]) for y in Y, x in X]
# Z2 = [pdf(MvNormal(unsafeMean(marginals[:m_2]),Matrix(Hermitian(inv(unsafeMean(marginals[:W_2]))))), [x,y]) for y in Y, x in X]
# Z3 = [pdf(MvNormal(unsafeMean(marginals[:m_3]),Matrix(Hermitian(inv(unsafeMean(marginals[:W_3]))))), [x,y]) for y in Y, x in X]
# contour!(X, Y, Z1)
# contour!(X, Y, Z2)
# contour!(X, Y, Z3)