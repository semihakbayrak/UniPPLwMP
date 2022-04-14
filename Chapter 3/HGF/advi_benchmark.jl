using Turing, Random, AdvancedVI, DelimitedFiles, DataFrames
using Plots
using BenchmarkTools
Random.seed!(0);

#Generate data
T = 400

vz, vy = 0.01, 0.1

z_data_0 = 0
z_data = [sin(pi/60) + sqrt(vz)*randn()]
x_data_0 = 0
x_data = [x_data_0 + sqrt(exp(z_data[1]))*randn()]
y_data = [x_data[1]+sqrt(vy)*randn()]
for t=2:T
    append!(z_data, sin(t*pi/60) + sqrt(vz)*randn())
    append!(x_data, x_data[end] + sqrt(exp(z_data[end]))*randn())
    append!(y_data, x_data[end]+sqrt(vy)*randn())
end

function run_advi()
    @model function HGF(m_z_t_min, v_z_t_min, m_x_t_min, v_x_t_min, y_t)
        model_vz, model_vy = 0.1, 0.1
        z_t_min ~ Normal(m_z_t_min, sqrt(v_z_t_min))
        z_t ~ Normal(z_t_min, sqrt(model_vz))
        x_t_min ~ Normal(m_x_t_min, sqrt(v_x_t_min))
        x_t ~ Normal(x_t_min, sqrt(exp(z_t)))
        y_t ~ Normal(x_t, sqrt(model_vy))
    end

    # Define values for prior statistics
    m_z_0, v_z_0 = 0.0, 1.0
    m_x_0, v_x_0 = 0.0, 1.0

    m_z_t_min, v_z_t_min = m_z_0, v_z_0
    m_x_t_min, v_x_t_min = m_x_0, v_x_0

    advi = ADVI(10, 4000)

    m_z = Vector{Float64}(undef, T)
    v_z = Vector{Float64}(undef, T)
    m_x = Vector{Float64}(undef, T)
    v_x = Vector{Float64}(undef, T)

    for t=1:T
        model = HGF(m_z_t_min, v_z_t_min, m_x_t_min, v_x_t_min, y_data[t])
        q = vi(model, advi);
        samples = rand(q,1000)
        m_z_t_min, v_z_t_min, m_x_t_min, v_x_t_min = mean(samples[2,:]), var(samples[2,:]), mean(samples[4,:]), var(samples[4,:])
        m_z[t], v_z[t], m_x[t], v_x[t] = m_z_t_min, v_z_t_min, m_x_t_min, v_x_t_min
    end
end

bench_advi = @benchmarkable run_advi()
run(bench_advi, samples=100, seconds=10000)