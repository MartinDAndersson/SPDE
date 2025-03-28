"""
    SPDE

Main module for the SPDE (Stochastic Partial Differential Equations) package. 
This module provides tools for numerically solving stochastic heat equations 
and estimating diffusion coefficients from solution paths.

The core functionality includes:
- Generating numerical solutions to SPDEs
- Applying differential operators for extracting diffusion coefficients
- Implementing partial integration methods
- Training machine learning models to estimate diffusion functions
"""
module SPDE
export experiment_3
export L_op
using LinearAlgebra, Random, Distributions, DifferentialEquations, LsqFit
using SparseArrays
using LaTeXStrings, LoopVectorization, DataFrames,CSV, JLD2, FileIO
import StatsBase,ProgressLogging
using StatsBase
using MLJ
using DrWatson
using NearestNeighborModels
#using ScikitLearn.GridSearch: GridSearchCV
using ProgressMeter
using Trapz
KNN = MLJ.@load KNNRegressor
NeuralNetworkRegressor = MLJ.@load NeuralNetworkRegressor pkg=MLJFlux
RandomForestRegressor = MLJ.@load RandomForestRegressor pkg=DecisionTree
#using PyCall
#const KNeighborsRegressor = PyNULL()
#function __init__()
#    @eval @sk_import neighbors: KNeighborsRegressor
#end
"""
    L_op(u, dt, dx)

Apply the differential operator to a solution matrix to extract information about the 
diffusion coefficient σ².

This operator implements: L[u] = ∂u/∂t - (1/2)∂²u/∂x²

# Arguments
- `u::Matrix{Float64}`: Solution matrix with dimensions (space, time)
- `dt::Float64`: Time step size
- `dx::Float64`: Space step size

# Returns
- `Lu::Matrix{Float64}`: Matrix with the operator applied at each valid point

This function uses first-order finite differences for time derivatives and 
second-order central differences for spatial derivatives.
"""
function L_op(u,dt,dx)
    #u=BigFloat.(u)
    #u = permutedims(u,(2,1))
    Lu = similar(u)
    nnx,nnt = size(u)
    @inbounds for t in 1:nnt-1
        @inbounds for x in 2:nnx-1
            t_diff = 1/dt*(u[x,t+1]-u[x,t])
            x_diff = (1/dx^2)*(u[x-1,t]-2*u[x,t]+u[x+1,t]) 
            Lu[x,t] = t_diff - 1/2 .* x_diff
        end
    end
    return Lu
end

"""
    L_op(u, dt, dx, xdiff, tdiff)

Apply the differential operator with custom stencil sizes.

This version allows for larger stencils in both space and time dimensions,
which can improve accuracy at the cost of reduced domain size.

# Arguments
- `u::Matrix{Float64}`: Solution matrix with dimensions (space, time)
- `dt::Float64`: Time step size
- `dx::Float64`: Space step size
- `xdiff::Int`: Number of spatial grid points to use in the difference stencil
- `tdiff::Int`: Number of time steps to use in the difference stencil

# Returns
- `Lu::Matrix{Float64}`: Matrix with the operator applied at each valid point
"""
function L_op(u,dt,dx,xdiff,tdiff)
    #u=BigFloat.(u)
    #u = permutedims(u,(2,1))
    Lu = similar(u)
    nnx,nnt = size(u)
    @inbounds for t in 1:nnt-1-tdiff
        @inbounds for x in 2+xdiff:nnx-1-xdiff
            t_d = 1/dt*(u[x,t+tdiff]-u[x,t])
            x_d = (1/dx^2)*(u[x-xdiff,t]-2*u[x,t]+u[x+xdiff,t]) 
            Lu[x,t] = t_d - 1/2 .* x_d
        end
    end
    return Lu
end

"""
    downsample_matrix(matrix, fx, fy)

Downsample a matrix by averaging over fx × fy blocks.

# Arguments
- `matrix::Matrix{T}`: Input matrix to be downsampled
- `fx::Int`: Downsampling factor in x dimension
- `fy::Int`: Downsampling factor in y dimension

# Returns
- `downsampled::Matrix{Float64}`: Downsampled matrix

This function is useful for reducing the resolution of solution fields while
preserving the overall structure through local averaging.
"""
function downsample_matrix(matrix, fx, fy)
    # Downsample a matrix by averaging over fx by fy blocks
    Nx, Ny = size(matrix)
    new_Nx, new_Ny = ceil(Int, Nx/fx), ceil(Int, Ny/fy)
    downsampled = zeros(new_Nx, new_Ny)
    
    for i in 1:new_Nx
        for j in 1:new_Ny
            x_start, y_start = (i-1)*fx + 1, (j-1)*fy + 1
            x_end, y_end = min(i*fx, Nx), min(j*fy, Ny)
            downsampled[i, j] = mean(matrix[x_start:x_end, y_start:y_end])
        end
    end
    
    return downsampled
end

"""
    get_drift_diffusion()

Create the drift and diffusion functions for stochastic heat equation.

# Returns
- `(heat_drift!, heat_diffusion_noise!)`: A tuple of two functions:
  - `heat_drift!`: Implements the deterministic part (∂²u/∂x²)
  - `heat_diffusion_noise!`: Implements the stochastic part (σ(u) dW/dt)

These functions are compatible with the DifferentialEquations.jl interface for SDEs and
have the correct boundary conditions (zero at boundaries).

The drift function implements the heat operator with fixed boundary conditions:
∂u/∂t = (1/2)∂²u/∂x²

The diffusion function implements the state-dependent noise term:
σ(u) * sqrt(1/dx)
"""
function get_drift_diffusion() 
	function heat_drift!(du, u, p, t)
		@inbounds begin
		du[1] = 0.0
		du[end] = 0.0
		end
		M = length(du)
		dx,σ = p
		@inbounds for i in 2:M-1
			du[i] = 0.5 .* (u[i-1] - 2u[i] + u[i+1])/(dx^2)
		end
	end
	function heat_diffusion_noise!(du, u, p, t)
		dx,σ = p
		@inbounds for i in 2:length(u)-1
			du[i] = σ(u[i])*sqrt(1/dx)
		end
		@inbounds begin
            du[1] = 0
            du[end] = 0
		end
	end
	return heat_drift!,heat_diffusion_noise!
end


function partial_integration(solution,dt,dx,x_quot,t_quot,eps)
    nx, nt = size(solution)
    
    # Calculate new step sizes
    new_dx = x_quot * dx
    new_dt = t_quot * dt
    # Calculate window sizes in grid points
    x_eps = div(eps, dx) |> Int  # Integer division (no need for |> Int)
    t_eps = div(eps, dt) |> Int
    x_idx = 1:x_quot:nx#[i for i in 1:new_nx]*x_quot
    t_idx = 1:t_quot:nt#[t for t in 1:new_nt]*t_quot
    new_sol = @view solution[x_idx,t_idx]#downsample_matrix(solution,x_quot,t_quot)
    Lu = L_op(new_sol,new_dt,new_dx) .* 1/(sqrt(2)*eps)   #.* dx .* dt #.* new_dx .* new_dt #.* sqrt(dx*dt) 
    buffer = 1#2^13 ÷ t_quot
    x_len,t_len = size(Lu)
    #time_startup = 2^15 ÷ t_quot
    max_x_points = (x_len)-x_eps-1 -(x_eps+1)
    num_x_samples = min(10,max_x_points)
    total_samples = min(50000,t_len*num_x_samples)
    @show factor = t_len*num_x_samples/total_samples |> x-> ceil(Int,x)
    results = Channel{Tuple}(Inf)
    Threads.@threads for t in 1:factor:t_len-t_eps-10
        rand_x = sample(x_eps+32:x_len-x_eps-32,num_x_samples,replace=false) #+ -2
        for i in 1:num_x_samples
            #x = rand(x_eps+1:x_len-x_eps-1)
            x=rand_x[i]
            integrated_view = view(Lu, x-x_eps:x+x_eps, t:t+t_eps) # right now not shifted to -1 to compensate
            l1,l2 = size(integrated_view)
            rx = range(0,dx*(l1),length=l1)
            rt = range(0,dt*(l2),length=l2)
            integrated=trapz((rx,rt),integrated_view)^2
            #integrated = sum(integrated_view)^2
            #plotln(integrtated)
            #integrated = sum(Lu[x:x+x_eps-1,t:t+t_eps-1]).^2  # with time
            #integrated = sum(Lu[x:x+x_eps-1,t]).^2 .* new_dt
            u=new_sol[x,t]
            put!(results, (u, integrated))
            #push!(df,(u,integrated))
        end
    end
    close(results)
    collected_results = collect(results)
    df = DataFrame(:x=>first.(collected_results),:y=>last.(collected_results))
    #zipped=zip(new_sol[buffer:end-buffer,ks:end],Lu[buffer:end-buffer,ks:end]) |> collect |> x->reshape(x,:)
    return df
end

"""
    partial_integration(solution, dt, dx, x_quot, t_quot, eps; num_samples=50000)

Perform partial integration of the SPDE operator over spatial-temporal regions to estimate the diffusion coefficient.

This function implements the core estimation technique described in the theory document. It:
1. Computes the differential operator L applied to the solution
2. Integrates L over small regions of size approximately 'eps'
3. Samples points to create a dataset of (u, integrated L²) pairs for regression

# Arguments
- `solution`: Matrix representing the solution where rows are space points and columns are time points
- `dt`: Time step size of the solution
- `dx`: Space step size of the solution
- `x_quot`: Space downsampling factor (use 1 for no downsampling)
- `t_quot`: Time downsampling factor (use 1 for no downsampling)
- `eps`: Size of the integration region (in original units, not grid points)
- `num_samples`: Maximum number of samples to collect (default: 50000)

# Returns
- `DataFrame`: A data frame with columns `:x` (solution values) and `:y` (corresponding squared integrals)

# Notes
- This implementation fixes scaling issues related to downsampling
- The integration is performed within a region of size approximately `eps` around each sampled point
- Stratified sampling is used to ensure good coverage of the solution value range
- The boundary regions are excluded from sampling to avoid edge effects
"""


function partial_integration(solution, dt, dx, x_quot, t_quot, eps; num_samples=50000)
    # Input validation
    nx, nt = size(solution)
    if nx < 2 || nt < 2
        throw(ArgumentError("Solution matrix must have at least 2 rows and 2 columns"))
    end
    if eps <= 0 || dt <= 0 || dx <= 0 || x_quot <= 0 || t_quot <= 0
        throw(ArgumentError("All step sizes and quotients must be positive"))
    end
    
    # Calculate new step sizes after downsampling
    new_dx = x_quot * dx
    new_dt = t_quot * dt
    
    # Calculate window sizes in grid points for the DOWNSAMPLED grid
    # This ensures the physical window size is consistent regardless of downsampling
    x_eps = max(1, round(Int, eps / new_dx))  # Window size in GRID POINTS
    t_eps = max(1, round(Int, eps / new_dt))
    
    # Downsample the solution
    x_idx = 1:x_quot:nx
    t_idx = 1:t_quot:nt
    new_sol = @view solution[x_idx, t_idx]
    
    # Calculate the physical size of the integration window
    window_width_x = (2 * x_eps + 1) * new_dx  # Physical width in space
    window_width_t = (t_eps + 1) * new_dt      # Physical duration in time
    window_volume = window_width_x * window_width_t
    
    # Calculate correct scaling factor based on the mathematical formula in the paper
    # For a spatial covariance Γ, m(ε) = sqrt(ε∫∫Γ(x-y)dxdy)
    # This is a simplified form matching the paper's notation
    scaling_factor = 1 / sqrt(window_volume)
    
    # Compute Lu with proper scaling
    Lu = L_op(new_sol, new_dt, new_dx) .* scaling_factor
    
    # Get dimensions of downsampled solution
    x_len, t_len = size(Lu)
    
    # Define buffer zones to avoid boundary effects
    x_buffer = x_eps + 10  # Extra padding to avoid boundary effects
    t_buffer = t_eps + 10  # Extra padding for time dimension
    
    # First pass: Get a sense of the distribution of u values
    # Sample a smaller subset to analyze the distribution
    sample_fraction = 0.1  # Use 10% for analysis
    pilot_samples = min(5000, round(Int, num_samples * sample_fraction))
    
    # Create exploration grid covering the valid domain
    valid_x_range = (x_buffer+1):(x_len-x_buffer)
    valid_t_range = 1:(t_len-t_buffer)
    
    # Ensure we have enough points in the domain for sampling
    if isempty(valid_x_range) || isempty(valid_t_range)
        throw(ArgumentError("Integration region (eps) is too large for the solution domain after downsampling"))
    end
    
    # Create systematic grid for exploration
    pilot_u_values = Float64[]
    pilot_points = Tuple{Int,Int}[]
    
    # Sample some points systematically to understand the distribution
    x_step = max(1, div(length(valid_x_range), ceil(Int, sqrt(pilot_samples))))
    t_step = max(1, div(length(valid_t_range), ceil(Int, sqrt(pilot_samples))))
    
    for x in valid_x_range[1:x_step:end]
        for t in valid_t_range[1:t_step:end]
            push!(pilot_u_values, new_sol[x,t])
            push!(pilot_points, (x,t))
            length(pilot_u_values) >= pilot_samples && break
        end
        length(pilot_u_values) >= pilot_samples && break
    end
    
    # Analyze distribution and create strata for better sampling
    n_strata = 10  # Number of equal-sized bins
    sorted_indices = sortperm(pilot_u_values)
    strata_size = max(1, length(pilot_u_values) ÷ n_strata)
    strata_bounds = Float64[]
    
    for i in 1:n_strata-1
        idx = sorted_indices[min(i * strata_size, length(sorted_indices))]
        push!(strata_bounds, pilot_u_values[idx])
    end
    pushfirst!(strata_bounds, -Inf)
    push!(strata_bounds, Inf)
    
    # Second pass: Sample from each stratum to ensure good coverage
    results = Channel{Tuple{Float64, Float64}}(Inf)
    samples_per_stratum = num_samples ÷ n_strata
    
    # Prepare all valid points for sampling
    all_valid_points = Vector{Tuple{Int,Int}}()
    for x in valid_x_range
        for t in valid_t_range
            push!(all_valid_points, (x,t))
        end
    end
    
    # Shuffle to randomize before stratification
    shuffle!(all_valid_points)
    
    # Process each stratum in parallel
    Threads.@threads for stratum in 1:n_strata
        lower_bound = strata_bounds[stratum]
        upper_bound = strata_bounds[stratum+1]
        
        stratum_count = 0
        for (x, t) in all_valid_points
            # Check if this point's u value falls in our stratum
            u_val = new_sol[x,t]
            if lower_bound < u_val <= upper_bound
                # Make sure we can create a full integration window
                if x-x_eps < 1 || x+x_eps > x_len || t+t_eps > t_len
                    continue
                end
                
                # Extract the integration region
                integrated_view = view(Lu, (x-x_eps):(x+x_eps), t:(t+t_eps))
                l1, l2 = size(integrated_view)
                
                # Create CORRECT grid spacings for integration
                rx = range(0, new_dx*(l1), length=l1)  # Using new_dx
                rt = range(0, new_dt*(l2), length=l2)  # Using new_dt
                
                # Perform the integration and square the result
                integrated = trapz((rx,rt), integrated_view)^2
                
                # Add to results
                put!(results, (u_val, integrated))
                stratum_count += 1
                
                # Stop when we have enough samples for this stratum
                stratum_count >= samples_per_stratum && break
            end
        end
    end
    
    # Close the channel and collect results
    close(results)
    collected_results = collect(results)
    
    # Check if we have enough data points
    if length(collected_results) < 10
        @warn "Very few data points collected ($(length(collected_results))). Check your parameters."
    end
    
    # Return as DataFrame
    return DataFrame(:x => first.(collected_results), :y => last.(collected_results))
end


"""
    partial_integration(solution, dt, dx, x_quot, t_quot, eps)

Perform partial integration of the SPDE operator over spatial-temporal regions to estimate the diffusion coefficient.

This function implements the core estimation technique described in the theory document. It:
1. Computes the differential operator L applied to the solution
2. Integrates L over small regions of size approximately 'eps'
3. Samples points to create a dataset of (u, integrated L²) pairs for regression

# Arguments
- `solution`: Matrix representing the solution where rows are space points and columns are time points
- `dt`: Time step size of the solution
- `dx`: Space step size of the solution
- `x_quot`: Space downsampling factor (use 1 for no downsampling)
- `t_quot`: Time downsampling factor (use 1 for no downsampling)
- `eps`: Size of the integration region (in original units, not grid points)

# Returns
- `DataFrame`: A data frame with columns `:x` (solution values) and `:y` (corresponding squared integrals)

# Notes
- The integration is performed within a region of size approximately `eps` around each sampled point
- The boundary regions are excluded from sampling to avoid edge effects
- The function uses parallel processing to speed up computation
"""
# function partial_integration(solution, dt, dx, x_quot, t_quot, eps; 
#                             max_samples::Int=50000, 
#                             points_per_time::Int=10,
#                             boundary_margin::Int=32)
#     # Input validation
#     nx, nt = size(solution)
#     # Calculate new step sizes after downsampling
#     new_dx = x_quot * dx
#     new_dt = t_quot * dt
    
#     # Calculate window sizes in grid points for the DOWNSAMPLED grid
#     x_eps = max(1, round(Int, eps / new_dx))
#     t_eps = max(1, round(Int, eps / new_dt))
    
#     # Downsample the solution
#     x_idx = 1:x_quot:nx
#     t_idx = 1:t_quot:nt
#     new_sol = @view solution[x_idx, t_idx]
    
#     # Calculate the physical size of the integration window
#     window_width_x = 2 * x_eps * new_dx
#     window_width_t = t_eps * new_dt
#     scaling_factor = 1 / sqrt(window_width_x * window_width_t)
#     num_points_x = 2 * x_eps + 1
#     num_points_t = t_eps + 1
    
#     # Calculate the normalization denominator squared (variance sum for sigma=1)
#     m_eps_squared_discrete = num_points_x * num_points_t * new_dx * new_dt
#     #scaling_factor = 1/sqrt(m_eps_squared_discrete)
#     # Log the integration parameters
#     @debug "Integration parameters" x_eps t_eps new_dx new_dt
    
#     # Get downsampled solution if quotients > 1
#     new_sol = if x_quot > 1 || t_quot > 1
#         x_idx = 1:x_quot:nx
#         t_idx = 1:t_quot:nt
#         @view solution[x_idx, t_idx]
#     else
#         @view solution[:, :]
#     end
    
#     # Calculate the differential operator with appropriate scaling
#     Lu = L_op(new_sol, new_dt, new_dx) .* scaling_factor
    
#     # Determine dimensions and sampling parameters
#     x_len, t_len = size(Lu)
    
#     # Ensure we have enough points in the domain for sampling
#     effective_x_domain = x_len - 2*(x_eps + boundary_margin)
#     if effective_x_domain < 1
#         throw(ArgumentError("Integration region (eps) is too large for the solution domain"))
#     end
    
#     # Calculate number of points to sample per time step
#     num_x_samples = min(points_per_time, effective_x_domain)
    
#     # Calculate total desired samples and determine sampling frequency
#     total_desired_samples = min(max_samples, t_len * num_x_samples)
#     sample_every_n_timesteps = ceil(Int, (t_len * num_x_samples) / total_desired_samples)
    
#     # Initialize channel for collecting results
#     results = Channel{Tuple{Float64, Float64}}(Inf)
    
#     # Parallel processing of time steps
#     #Threads.@threads
#     Threads.@threads for t in 1:sample_every_n_timesteps:t_len-t_eps-1
#         # Sample spatial points at this time step
#         safe_min = x_eps + boundary_margin
#         safe_max = x_len - x_eps - boundary_margin
        
#         # Skip if we don't have enough space to sample
#         if safe_min >= safe_max
#             @debug "Skipping time step $t due to insufficient space"
#             continue
#         end
        
#         # Sample points in space
#         rand_x = sample(safe_min:safe_max, num_x_samples, replace=false)
        
#         # Process each spatial point
#         for x in rand_x
#             # Extract view for integration region
#             integrated_view = view(Lu, (x-x_eps):(x+x_eps), t:(t+t_eps))
            
#             # Create appropriate ranges for integration
#             l1, l2 = size(integrated_view)
#             rx = range(0, new_dx*(l1), length=l1)
#             rt = range(0, new_dt*(l2), length=l2)
            
#             # Perform integration and square the result
#             integrated = trapz((rx, rt), integrated_view)^2
            
#             # Get corresponding solution value
#             u_value = new_sol[x, t]
            
#             # Add to results
#             put!(results, (u_value, integrated))
#         end
#     end
    
#     # Close the channel and collect results
#     close(results)
#     collected_results = collect(results)
    
#     # Return as DataFrame
#     return DataFrame(:x => first.(collected_results), :y => last.(collected_results))
# end



"""
    monte_carlo_error(mach, truth, domain, N, loss)

Calculate the error between a machine learning model's predictions and the ground truth
using Monte Carlo integration.

# Arguments
- `mach::Machine`: Trained MLJ machine
- `truth::Function`: Ground truth function for σ²(u)
- `domain::Tuple{Float64,Float64}`: Range (low, high) to evaluate over
- `N::Int`: Number of Monte Carlo samples
- `loss::Function`: Loss function to apply to differences

# Returns
- `error::Float64`: Average loss over N random points in the domain
"""
function monte_carlo_error(mach,truth,domain,N,loss)
	fun = mach_to_func(mach)
	low=domain[1]
	up = domain[2]
	points = rand(Distributions.Uniform(low,up),N)
	error = 0.
	for x in points
		error += loss(fun(x)-truth(x))
	end
	return error/N
end

"""
    get_best_machine(machs, truth, domain, N, loss)

Select the best machine learning model from a collection based on error evaluation.

# Arguments
- `machs::Vector{Machine}`: List of trained MLJ machines
- `truth::Function`: Ground truth function for σ²(u)
- `domain::Tuple{Float64,Float64}`: Range (low, high) to evaluate over
- `N::Int`: Number of Monte Carlo samples
- `loss::Function`: Loss function to apply to differences

# Returns
- `(best_mach, best_error)::Tuple{Machine,Float64}`: The best machine and its error
"""
function get_best_machine(machs,truth,domain,N,loss)
	best_error = Inf
	best_mach = machs[1]
	for mach in machs
		#func = mach_to_func(mach)
		#p=1.
		error = monte_carlo_error(mach,truth,domain,N,loss)
		#g(x,p) = abs(func(x)-truth(x))^2
		#prob = IntegralProblem(g, domain, p)
		#sol = solve(prob, QuadGKJL(),maxiters=10000)
		if error < best_error 
			best_error = error
			best_mach = mach
		end
	end
	return (best_mach,best_error)
end

"""
    mach_to_func(mach)

Convert an MLJ machine into a callable function for evaluation.

# Arguments
- `mach::Machine`: Trained MLJ machine

# Returns
- `est_wrapper::Function`: Function that takes a single value and returns the model prediction
"""
function mach_to_func(mach)
	est_wrapper(x) = MLJ.predict(mach,DataFrame(:x=>[x]))[1]
	return est_wrapper
end

"""
    get_all_losses(mach, truth, domain, N)

Calculate multiple error metrics for model evaluation.

# Arguments
- `mach::Machine`: Trained MLJ machine
- `truth::Function`: Ground truth function for σ²(u)
- `domain::Tuple{Float64,Float64}`: Range (low, high) to evaluate over
- `N::Int`: Number of Monte Carlo samples

# Returns
- `(l1, l2)::Tuple{Float64,Float64}`: L1 (absolute) and L2 (squared) errors
"""
function get_all_losses(mach,truth,domain,N)
	losses = Dict("l1"=>x->abs(x),"l2"=>x->abs(x)^2)
	l1 = monte_carlo_error(mach,truth,domain,N,losses["l1"])
	l2 = monte_carlo_error(mach,truth,domain,N,losses["l2"])
	return (l1,l2)
end


"""
    main_exp(spde_params)

Run the main experiment for diffusion coefficient estimation.

# Arguments
- `spde_params::Dict`: Dictionary containing experiment parameters:
  - `xquot::Int`: Spatial downsampling factor
  - `tquot::Int`: Temporal downsampling factor
  - `epsilon::Float64`: Size of integration window
  - `sigma::String`: Key for the diffusion function to use

# Returns
- `(mach, fulld, truth, df)::Tuple`: A tuple containing:
  - `mach::Machine`: Trained machine learning model
  - `fulld::Dict`: Dictionary with original params plus error metrics
  - `truth::Function`: Ground truth diffusion function 
  - `df::DataFrame`: Data used for training

This function implements the full estimation pipeline for a given diffusion function,
generating multiple SPDE solutions, applying partial integration, training a model,
and evaluating its accuracy.
"""
function main_exp(spde_params)
    sigma_1(x) =  1.
    sigma_2(x) = 0.5*sin(2*6*pi/2*x)#0.5x
    sigma_3(x) = 0.5*sin(6*x/(2*pi))
    sigma_4(x) = 0.5x
    sigmas = Dict("sigma1"=>sigma_1,"sigma2"=>sigma_2,"sigma3"=>sigma_3,"sigma4"=>sigma_4)
    @unpack xquot, tquot, epsilon, sigma = spde_params
    σ = sigmas[sigma]  #sigmas[sigma]
    L=1 # right boundary of space
    tmax = 0.3;
	#σ = x-> 0.5*sin(2pi*x/6)
    h=2^9
    nx = h;
	nt= 2^18
    dx = L/(nx-1); 
	dt = tmax/(nt-1);
    tmax = 0.3; 
    truth(x) = σ(x)^2
    #ϵ = 2dx
	#figname = "0.5x"
	samples = 1;
    #solution = generate_solution(σ)
    #println(size(solution))
    sol = @view(solution[:,:])
    df = DataFrame(:x=>Float64[],:y=>Float64[])
    while size(df,1) < 20000
        solution = generate_solution(σ)
        df_partial=SPDE.partial_integration(sol,dt,dx,xquot,tquot,epsilon*dx)
        df = vcat(df,df_partial)
    end
    max_size_df = min(50000,size(df,1))
    rand_rows = sample(1:size(df,1),max_size_df,replace=false)
    df=df[rand_rows,:]
    println(size(df,1))
    mach = train_tuned(df)
    N=10000
    domain=(2,6)
    l1,l2 = get_all_losses(mach,truth,domain,N)
    fulld = copy(spde_params)
    fulld["l1"] = l1
    fulld["l2"] = l2
    return mach,fulld,truth,df
end

"""
    paper_exp(solution, spde_params, σ)

Run the experiment with parameters used in the research paper.

# Arguments
- `solution::Array`: Initial SPDE solution to start with
- `spde_params::Dict`: Dictionary containing experiment parameters
- `σ::Function`: Diffusion coefficient function

# Returns
- `(mach, fulld)::Tuple`: A tuple containing:
  - `mach::Machine`: Trained machine learning model
  - `fulld::Dict`: Dictionary with original params plus error metrics and domain bounds

This version of the experiment is tuned specifically for reproducing results 
in the research paper, with careful domain boundary handling.
"""
function paper_exp(solution,spde_params,σ)
    @unpack xquot, tquot, epsilon, sigma = spde_params
    #σ(x) = sin(x)
    L=1; 
    tmax = 0.3;
    h=2^9
    nx = h;
    nt= 2^18
    dx = L/(nx-1); 
    dt = tmax/(nt-1); 
    df = DataFrame(:x=>Float64[],:y=>Float64[])
    sol = @view(solution[:,:])
    df_partial=SPDE.partial_integration(sol,dt,dx,xquot,tquot,epsilon)
    df = vcat(df,df_partial)
    #σ(x) = sin(x)
    while size(df,1) < 20000
        solution = generate_solution(σ)
        sol = @view(solution[:,:])
        df_partial=SPDE.partial_integration(sol,dt,dx,xquot,tquot,epsilon)
        df = vcat(df,df_partial)
    end
    max_size_df = min(50000,size(df,1))
    rand_rows = sample(1:size(df,1),max_size_df,replace=false)
    df=df[rand_rows,:]
    mach = train_tuned(df)
    lower_bound = quantile(df.x, 0.05)
	upper_bound = quantile(df.x, 0.95)
	# Filter the array to keep only the middle 90%
	filtered_data = filter(x -> x >= lower_bound && x <= upper_bound, df.x)
	umax = maximum(filtered_data)
	umin = minimum(filtered_data)
    truth(x) = σ(x)^2
    N=10000
    domain=(umin,umax)
    l1,l2 = get_all_losses(mach,truth,domain,N)
    println(l1,l2)
    fulld = copy(spde_params)
    fulld["l1"] = l1
    fulld["l2"] = l2
    fulld["lb"] = umin
    fulld["ub"] = umax
    return mach,fulld
end

"""
    generate_solution(σ, nx, nt)

Generate a numerical solution to a stochastic partial differential equation.

# Arguments
- `σ`: Diffusion coefficient function
- `nx`: Number of spatial grid points
- `nt`: Number of time grid points

# Returns
- Solution of the SPDE as an ensemble
"""
function generate_solution(σ, nx, nt;
    L::Float64=1.0, t_buffer::Float64=0.1, t_end::Float64=0.4,
    u0_val::Float64=2π, solver=EM(), ensemble_size::Int=1)
    
    # Domain parameters
    L = 1.0
    buffer_t = 0.1
    t_max = 0.4 + buffer_t
    
    # Grid spacing
    dx = L / (nx - 1)
    dt = (t_max - buffer_t) / (nt - 1)
    
    # Initial condition (with boundary conditions)
    u_begin = 2π * ones(nx)
    u_begin[1] = 0
    u_begin[end] = 0
    
    # Set up SDE problem
    drift, diffusion = SPDE.get_drift_diffusion()
    jac_prototype = Tridiagonal([1.0 for _ in 1:nx-1], 
                               [-2.0 for _ in 1:nx], 
                               [1.0 for _ in 1:nx-1]) |> sparse
    sde_func = SPDE.SDEFunction(drift, diffusion; jac_prototype=jac_prototype)
    
    # Time settings
    time_range = (0.0, t_max)
    t_save_points = buffer_t:1*dt:t_max
    
    # Parameters for the drift and diffusion functions
    p = (dx, σ)
    
    # Create and solve the ensemble problem
    prob = SDEProblem(sde_func, u_begin, time_range, p)
    ensemble_prob = EnsembleProblem(prob)
    
    # Solve using Euler-Maruyama method
    solution = solve(
        ensemble_prob,
        EM(),  # Using Euler-Maruyama algorithm
        dtmax=dt,
        dt=dt,
        trajectories=1,
        saveat=t_save_points,
        progress=true,
        maxiters=1e7
    )
    
    return solution
end
# function generate_solution(σ,nx,nt)
#     #algo = ImplicitEM(linsolve = KLUFactorization())
#     #algo=SRIW1()
#     algo = SROCK1()
#     algo=EM()
#     L=1; 
#     buffert=0.1;
# 	tmax = 0.4+buffert;
# 	#h=2^8
# 	#nx = h;
# 	#nt= 2^18
# 	dx = L/(nx-1); 
# 	dt = (tmax-buffert)/(nt-1); 
#     max_ϵ = 32dx
# 	u_begin = 2*pi*ones(nx); u_begin[1] = 0; u_begin[end] = 0;
# 	drift,diff = SPDE.get_drift_diffusion()
#     t0=tmax;
# 	jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
# 	SDE_sparse = SPDE.SDEFunction(drift,diff;jac_prototype=jac_prot)
# 	#t0=tmax;
# 	#ϵ = 32dx
# 	time_range = ((0.0,t0+max_ϵ+dt)) # fixa detta
#     t_idxs = buffert:1*dt:tmax+buffert #changed dt here
#     #t_idxs = #range(0.05,tmax,length=nt)
#     p=(dx,σ)
#     #x_idxs = 1:2:
#     E=EnsembleProblem(SDEProblem(SDE_sparse,u_begin,time_range,p))
#     solution=solve(E,dtmax=dt,dt=dt,trajectories=10,saveat=t_idxs,progress=true,algo,maxiters=1e7,save_idxs=[2^i for i in 1:7])
#     return solution
# end
"""
    train_tuned(df)

Train a k-nearest neighbor regression model with hyperparameter tuning.

# Arguments
- `df::DataFrame`: DataFrame with columns :x (feature values) and :y (target values)

# Returns
- `mach::Machine`: Trained MLJ machine with optimized hyperparameters

This function creates a tuned k-nearest neighbors model with cross-validation,
optimizing the number of neighbors and leaf size for best performance.
The model is trained to predict the squared diffusion coefficient σ²(u)
from solution values u.

The hyperparameter tuning is parallelized using Julia's threading capabilities
for efficient execution on multi-core systems.
"""
function train_tuned(df)
	df_filtered = df #filter(row -> row.y > 0.1, df)
	#df_train,df_test = partition(df_filtered  ,0.8,rng=123)
	x_data = select(df_filtered,1)
	y_data = df_filtered.y #.* nx/t_quot
    max_nbors = min(1500,length(df.x))
	knn=KNN()
	knn_r1=range(knn,:K,lower=5,upper=max_nbors)
	knn_r2=range(knn,:leafsize,lower=50,upper=100)
	knn_r3 = range(knn,:weights,values=[NearestNeighborModels.Dudani(),NearestNeighborModels.Uniform(),
	NearestNeighborModels.DualD(),NearestNeighborModels.DualU()])
	knn_tuned = TunedModel(model=knn,
	resampling=CV(nfolds=5),
	tuning=Grid(goal=10),
	range=[knn_r1],measure=LPLoss(;p=2),
	acceleration=CPUThreads(),
	acceleration_resampling=CPUThreads());
    RF = RandomForestRegressor()
    NNR=NeuralNetworkRegressor()
    mach = machine(knn_tuned,x_data,y_data)
    fit!(mach)
    return mach
end

end