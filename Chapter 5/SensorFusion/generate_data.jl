using LinearAlgebra, Plots, Random, DelimitedFiles, DataFrames, Distributions

Random.seed!(1234);

# Newtonian dynamics assumption
A = [1. 0. 1. 0.; 0. 1. 0. 1.; 0. 0. 1. 0.; 0. 0. 0. 1.]
B = [1. 0. 0. 0.; 0. 1. 0. 0.]
qc = 1.
Q = qc*[1/3 0 1/2 0; 0 1/3 0 1/2; 1/2 0 1 0; 0 1/2 0 1]
L = cholesky(Q).L
R = diagm(0=>ones(3))

T = 30
global position = 2*ones(2,T)
global velocity = [-2.5,1.6]

for t=2:T
    state = [position[:,t-1];velocity]
    state = A*state + L*randn(4)
    global position[:,t] = state[1:2]
    global velocity = state[3:4]
end

# sensor locations
sensor1 = [5.,5.]
sensor2 = [-40., -10.]
sensor3 = [0.,20.]

observation = []

disturbance1 = Normal(0,sqrt(1.0))
disturbance2 = Normal(0,sqrt(1.0))
disturbance3 = Normal(0,sqrt(1.0))

for t=1:T
    o1 = sqrt(sum((position[:,t]-sensor1).^2)) + rand(disturbance1)
    o2 = sqrt(sum((position[:,t]-sensor2).^2)) + rand(disturbance2)
    o3 = sqrt(sum((position[:,t]-sensor3).^2)) + rand(disturbance3)
    obs = [o1,o2,o3]
    push!(observation, obs)
end

# Save observation and sensor locations
writedlm("SensorFusion/sensors.txt", [sensor1, sensor2, sensor3])
writedlm("SensorFusion/observation.txt", observation)
writedlm("SensorFusion/position.txt", position)

# # Visualization of sensors and true discrete positions of the moving object
# plot(position[1,:],position[2,:],color=:redsblues,legend=:topleft, label="Interpolated trajectory")
# plot!(position[1,:],position[2,:], seriestype = :scatter,color=:redsblues, label="True positions")
# plot!([sensor1[1],sensor2[1],sensor3[1]],[sensor1[2],sensor2[2],sensor3[2]],seriestype = :scatter,color=:black, markersize=5, markershape=:square, label="sensors")