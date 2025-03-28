using DrWatson
@quickactivate "SPDE"  # Activate the project

using SPDE
using Plots
using StatsBase
using Random
using MLJ
using Revise

# Set random seed for reproducibility
Random.seed!(42)

#---------------------------------------------------
# Test parameters
#---------------------------------------------------
# Define a known diffusion function
# A fun "double bump" diffusion function
σ(x) = 0.2 + 0.4*exp(-16*(x-0.3)^2) + 0.3*exp(-14*(x-0.7)^2)
truth(x) = σ(x)^2  # The squared diffusion function we want to estimate

# SPDE simulation parameters
L = 1          # Spatial domain [0,L]
tmax = 0.3     # Time domain [0,tmax]
nx = 2^9       # Number of spatial grid points
nt = 2^18      # Number of time grid points
dx = L/(nx-1)  # Spatial grid size
dt = tmax/(nt-1)  # Time grid size

# Estimation parameters
x_quot = 1     # Spatial downsampling factor
t_quot = 1     # Temporal downsampling factor
epsilon = 4*dx   # Size of integration region

println("Generating SPDE solution...")
# Generate solution to the SPDE
solution = SPDE.generate_solution(σ, nx, nt)
sol = solution[1]#[1]
println("Computing estimator integral...")
# Compute the integrals for estimation
df = SPDE.partial_integration(sol, dt, dx, x_quot, t_quot, epsilon)

println("Dataset size: $(size(df, 1)) points")

# Train the estimator
println("Training regression model...")
mach = SPDE.train_tuned(df)

# Convert the model to a function
est_func = SPDE.mach_to_func(mach)
# Get the 5th percentile of the x values
lower = quantile(df.x, 0.05)
println("5th percentile of df.x: $lower")
upper = quantile(df.x, 0.95)
println("95th percentile of df.x: $upper")
y_lower = quantile(df.y, 0.05)
y_upper = quantile(df.y, 0.95)
y_bounds=(y_lower, y_upper)
# Calculate error metrics
domain = (lower, upper)  # Domain for error calculation
N = 1000  # Number of points for Monte Carlo error estimation
l1, l2 = SPDE.get_all_losses(mach, truth, domain, N)
println("L1 error: $l1")
println("L2 error: $l2")

# Plot the results
x_range = range(lower, upper, length=200)
p = plot(x_range, truth.(x_range), label="True σ²(x) = (0.5x)²", 
         linewidth=2, legend=:topleft, xlabel="u", ylabel="σ²(u)")
plot!(p, x_range, est_func.(x_range), label="Estimated σ²(u)", 
      linewidth=2, linestyle=:dash,ylims=y_bounds)

# Add some sample points from the dataset

scatter!(p, df.x[1:min(500,end)], df.y[1:min(500,end)], 
         label="Sample points", alpha=0.3, markersize=3,xlims=domain)

# Calculate and display the average quotient to check scaling
true_sigma = (σ.(df.x)).^2
quot=true_sigma ./ df.y
# Calculate the average quotient filtering out extreme values
filtered_quot = filter(q -> q <= 3, quot)
avg_quot_filtered = mean(filtered_quot)
println("Average quotient (filtered, estimated/true): $avg_quot_filtered")
println("Filtered $(length(quot) - length(filtered_quot)) extreme values out of $(length(quot))")
x_eval = range(lower, upper, length=50) 
#est_func.(x_eval)  truth.(x_eval) #|> scatter
# Avoid regions with very small values
quot = est_func.(x_eval) ./ truth.(x_eval)
avg_quot = mean(quot)
println("Average quotient (estimated/true): $avg_quot")

# Add this information to the plot
annotate!(p, [(3, maximum(truth.(x_range))*0.8, 
               text("Avg. quotient: $(round(avg_quot, digits=2))", 10, :left))])

display(p)
savefig(p, "spde_estimator_test.png")