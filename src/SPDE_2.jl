module SPDE

# --- Exports ---
export L_op, get_drift_diffusion, generate_solution, partial_integration
export train_tuned, monte_carlo_error, get_best_machine, mach_to_func, get_all_losses
export run_estimation_pipeline

# --- Imports ---
using LinearAlgebra, Random, Distributions, DifferentialEquations
using SparseArrays
using DataFrames, CSV, JLD2, FileIO
using StatsBase: sample, mean, quantile
using MLJ
using NearestNeighborModels
using ProgressMeter: @showprogress
using Trapz: trapz

# --- ML Model Loading (MLJ convention) ---
# Fix: Use the correct MLJ model loading syntax
const KNNRegressor = MLJ.@load KNNRegressor pkg=NearestNeighborModels

# --- Constants ---
const DEFAULT_BOUNDARY_MARGIN = 32
const DEFAULT_POINTS_PER_TIME = 10
const DEFAULT_MAX_SAMPLES = 50000
const MIN_TRAINING_SAMPLES = 20000

# --- Core Functions ---

"""
    L_op(u, dt, dx)

Apply the discretized differential operator L[u] = ∂u/∂t - (1/2)∂²u/∂x².
"""
function L_op(u::Matrix{Float64}, dt::Float64, dx::Float64)
    Lu = similar(u)
    nnx, nnt = size(u)
    dx_sq_inv = 1.0 / (dx^2)
    dt_inv = 1.0 / dt

    # Initialize with zeros to ensure well-defined values
    Lu .= 0.0

    # Loop bounds exclude boundaries where stencil is not fully defined
    @inbounds for t in 1:nnt-1
        @inbounds for x in 2:nnx-1
            # Forward difference in time
            t_diff = dt_inv * (u[x, t+1] - u[x, t])
            # Central difference in space
            x_diff = dx_sq_inv * (u[x-1, t] - 2 * u[x, t] + u[x+1, t])
            Lu[x, t] = t_diff - 0.5 * x_diff
        end
    end
    return Lu
end

"""
    L_op(u, dt, dx, xdiff, tdiff)

Apply the differential operator using wider stencils.
"""
function L_op(u::Matrix{Float64}, dt::Float64, dx::Float64, xdiff::Int, tdiff::Int)
    # Basic input validation
    (xdiff < 1 || tdiff < 1) && throw(ArgumentError("xdiff and tdiff must be >= 1"))

    Lu = similar(u)
    Lu .= 0.0  # Initialize to ensure well-defined values
    
    nnx, nnt = size(u)
    dx_sq_inv = 1.0 / (dx^2)
    dt_inv = 1.0 / (tdiff * dt) # Adjusted for wider time step

    # Loop bounds account for the wider stencil
    @inbounds for t in 1:nnt-tdiff
        @inbounds for x in (1+xdiff):(nnx-xdiff)
            # Forward difference in time over `tdiff` steps
            t_d = dt_inv * (u[x, t+tdiff] - u[x, t])
            # Central difference in space over `xdiff` distance
            x_d = dx_sq_inv * (u[x-xdiff, t] - 2 * u[x, t] + u[x+xdiff, t])
            Lu[x, t] = t_d - 0.5 * x_d
        end
    end
    return Lu
end

"""
    get_drift_diffusion(nx::Int)

Create the drift and diffusion functions for the stochastic heat equation.
"""
function get_drift_diffusion(nx::Int)
    function heat_drift!(du::AbstractVector, u::AbstractVector, p::Tuple, t::Float64)
        dx, _ = p # σ_func not needed for drift
        dx_sq_inv = 1.0 / (dx^2)

        # Boundary conditions
        du[1] = 0.0
        du[nx] = 0.0

        # Interior points: Central difference for second derivative
        @inbounds for i in 2:nx-1
            du[i] = 0.5 * dx_sq_inv * (u[i-1] - 2 * u[i] + u[i+1])
        end
        
        return nothing
    end

    function heat_diffusion_noise!(du::AbstractVector, u::AbstractVector, p::Tuple, t::Float64)
        dx, σ_func = p
        # Noise scaling factor for finite difference/volume method: sqrt(1/dx)
        noise_scaling = sqrt(1.0 / dx)

        # Boundary conditions (no noise at boundaries)
        du[1] = 0.0
        du[nx] = 0.0

        # Interior points
        @inbounds for i in 2:nx-1
            du[i] = σ_func(u[i]) * noise_scaling
        end
        
        return nothing
    end

    return heat_drift!, heat_diffusion_noise!
end

"""
    generate_solution(σ_func::Function, nx::Int, nt::Int;
                      L::Float64=1.0, t_buffer::Float64=0.1, t_end::Float64=0.4,
                      u0_val::Float64=2π, solver=EM(), ensemble_size::Int=1)

Generate numerical solution(s) to the stochastic heat equation.
"""
function generate_solution(σ_func::Function, nx::Int, nt::Int;
                           L::Float64=1.0, t_buffer::Float64=0.1, t_end::Float64=0.4,
                           u0_val::Float64=2π, solver=EM(), ensemble_size::Int=1)

    (nx < 3) && throw(ArgumentError("nx must be at least 3 for spatial discretization."))
    (nt < 1) && throw(ArgumentError("nt must be at least 1."))
    (t_end <= 0 || L <= 0) && throw(ArgumentError("t_end and L must be positive."))
    (t_buffer < 0) && throw(ArgumentError("t_buffer cannot be negative."))

    # Grid spacing
    dx = L / (nx - 1)
    # Time step for saving points (after buffer)
    dt_save = (t_end > 0) ? t_end / (nt - 1) : 0.0 # Avoid division by zero if t_end=0
    # Total simulation time
    t_max_sim = t_end + t_buffer

    # Choose a simulation dt (heuristic, may need tuning based on stability)
    dt_sim = min(dt_save, 0.5 * dx^2) # Heuristic stability guidance
    (dt_sim <= 0) && (dt_sim = 1e-6) # Ensure dt is positive if dt_save was 0

    @info "Simulation setup" L nx dx t_end t_buffer nt dt_save t_max_sim dt_sim

    # Initial condition
    u_begin = fill(Float64(u0_val), nx)
    u_begin[1] = 0.0
    u_begin[nx] = 0.0

    # Get drift/diffusion functions
    drift!, diffusion! = get_drift_diffusion(nx)

    # SDE Problem setup
    p = (dx, σ_func) # Parameters
    time_span = (0.0, t_max_sim)
    # Optional: Sparse Jacobian for implicit methods
    jac_prototype = Tridiagonal(ones(nx-1), -2*ones(nx), ones(nx-1)) * (0.5 / dx^2) |> sparse
    jac_prototype[1, :] .= 0
    jac_prototype[end, :] .= 0 # Boundaries
    sde_func = SDEFunction(drift!, diffusion!; jac_prototype=jac_prototype) # Add jac if needed by solver

    prob = SDEProblem(sde_func, u_begin, time_span, p)

    # Define points where solution is saved
    saveat_points = range(t_buffer, t_max_sim, length=nt)

    # Solve
    if ensemble_size == 1
        solution = solve(prob, solver; dt=dt_sim, saveat=saveat_points, progress=true, maxiters=Int(1e7))
    else
        ensemble_prob = EnsembleProblem(prob)
        solution = solve(ensemble_prob, solver; dt=dt_sim, saveat=saveat_points,
                         trajectories=ensemble_size, progress=true, maxiters=Int(1e7))
    end

    return solution
end

"""
    partial_integration(solution_matrix, dt, dx, eps;
                        max_samples::Int=DEFAULT_MAX_SAMPLES,
                        points_per_time::Int=DEFAULT_POINTS_PER_TIME,
                        boundary_margin::Int=DEFAULT_BOUNDARY_MARGIN)

Perform partial integration of the SPDE operator L[u] over space-time regions.
"""
function partial_integration(solution_matrix, dt::Float64, dx::Float64, eps::Float64;
                             max_samples::Int=DEFAULT_MAX_SAMPLES,
                             points_per_time::Int=DEFAULT_POINTS_PER_TIME,
                             boundary_margin::Int=DEFAULT_BOUNDARY_MARGIN)

    # --- Input Validation ---
    nx, nt = size(solution_matrix)
    if nx < 2*boundary_margin + 3 || nt < 3 # Need space for stencil and margin
        throw(ArgumentError("Solution matrix dimensions ($nx, $nt) too small for boundary_margin ($boundary_margin) and operator stencil."))
    end
    if eps <= 0 || dt <= 0 || dx <= 0
        throw(ArgumentError("eps, dt, and dx must be positive."))
    end
    if max_samples <= 0 || points_per_time <= 0 || boundary_margin < 0
        throw(ArgumentError("max_samples, points_per_time must be positive, boundary_margin non-negative."))
    end

    # --- Setup ---
    # Calculate integration region half-width in grid points (center point + x_eps_grid on each side)
    x_eps_grid = round(Int, eps / dx)
    t_eps_grid = round(Int, eps / dt)

    # Ensure grid steps are at least 0 (minimum region is a single point)
    x_eps_grid = max(0, x_eps_grid)
    t_eps_grid = max(0, t_eps_grid)

    # Minimum and maximum valid indices for the *center* of the integration box
    min_x_center = 1 + boundary_margin + x_eps_grid
    max_x_center = nx - boundary_margin - x_eps_grid
    min_t_center = 1 + t_eps_grid # Need t-t_eps_grid to exist
    max_t_center = nt - 1 - t_eps_grid # Need t+t_eps_grid and also t+1 for L_op

    if min_x_center > max_x_center || min_t_center > max_t_center
         throw(ArgumentError("Integration region size (eps=$eps -> grid steps $x_eps_grid, $t_eps_grid) or boundary_margin ($boundary_margin) too large for the solution domain ($nx, $nt). No valid points to sample."))
    end

    @info "Partial Integration Setup" eps dx dt x_eps_grid t_eps_grid boundary_margin min_x_center max_x_center min_t_center max_t_center

    # --- Calculate Operator ---
    Lu = L_op(solution_matrix, dt, dx)

    # --- Apply Scaling ---
    scaling_factor = 1.0 / (sqrt(2.0) * eps) # As per original code
    Lu .*= scaling_factor

    # --- Sampling Strategy ---
    num_valid_x = max_x_center - min_x_center + 1
    num_valid_t = max_t_center - min_t_center + 1

    points_to_sample_per_time = min(points_per_time, num_valid_x) # Don't sample more x than available

    # Estimate total possible samples if sampling every valid time step
    total_potential_samples = num_valid_t * points_to_sample_per_time

    # Determine sampling frequency in time to reach max_samples
    target_samples = min(max_samples, total_potential_samples)
    if target_samples == 0 # Should not happen due to checks above, but be safe
         @warn "No valid sampling points found after setup."
         return DataFrame(x=Float64[], y=Float64[])
    end

    # Sample roughly every `sample_every_n_timesteps` time steps
    sample_every_n_timesteps = max(1, floor(Int, total_potential_samples / target_samples))

    @info "Sampling Strategy" num_valid_x num_valid_t points_to_sample_per_time total_potential_samples target_samples sample_every_n_timesteps

    # --- Parallel Integration & Sampling ---
    results_channel = Channel{Tuple{Float64, Float64}}(Inf) # Unbuffered might be slow if consumer lags

    # Use @showprogress for a progress bar if ProgressMeter is available
    Threads.@threads for t_center in min_t_center:sample_every_n_timesteps:max_t_center
        # Sample distinct spatial points for this time step
        # Ensure we sample *center* points within the valid range
        x_centers_sampled = sample(min_x_center:max_x_center, points_to_sample_per_time, replace=false)

        for x_center in x_centers_sampled
            # Define the integration view boundaries
            x_start = x_center - x_eps_grid
            x_end   = x_center + x_eps_grid
            t_start = t_center - t_eps_grid
            t_end   = t_center + t_eps_grid # Lu computed up to nt-1, t_center max is nt-1-t_eps_grid

            # Extract the view of Lu for integration
            integration_view = view(Lu, x_start:x_end, t_start:t_end)

            # Create physical coordinate ranges for trapz
            # Length is (end - start + 1)
            range_x = range( (x_start-1)*dx, (x_end-1)*dx, length=size(integration_view, 1) )
            range_t = range( (t_start-1)*dt, (t_end-1)*dt, length=size(integration_view, 2) )

            # Perform 2D integration using trapezoidal rule
            integral_val = trapz((range_x, range_t), integration_view)

            # Get the corresponding solution value at the center of the box
            u_value = solution_matrix[x_center, t_center]

            # Store the result (u_value, squared_integral)
            put!(results_channel, (u_value, integral_val^2))
        end
    end

    close(results_channel)

    # --- Collect Results ---
    collected_results = collect(results_channel)

    if isempty(collected_results)
        @warn "No results collected during partial integration."
        return DataFrame(x=Float64[], y=Float64[])
    else
        # Convert to DataFrame
        df = DataFrame(x = first.(collected_results), y = last.(collected_results))
        @info "Partial integration finished. Collected $(nrow(df)) samples."
        return df
    end
end

# --- ML & Evaluation Functions ---

"""
    monte_carlo_error(mach::Machine, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int, loss::Function)

Estimate the average `loss` between a trained MLJ machine's predictions and a ground truth function.
"""
function monte_carlo_error(mach, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int, loss::Function)
    low, up = domain
    (low >= up) && throw(ArgumentError("Domain must be (low, high) with low < high."))
    (N < 1) && throw(ArgumentError("N must be at least 1."))

    points_x = rand(Distributions.Uniform(low, up), N)
    # Preallocate DataFrame for potentially faster prediction with some models
    predict_df = DataFrame(x=points_x)
    predictions = MLJ.predict(mach, predict_df)

    total_loss = 0.0
    for i in 1:N
        pred = predictions[i]
        true_val = truth_func(points_x[i])
        total_loss += loss(pred, true_val)
    end

    return total_loss / N
end

"""
    get_best_machine(machs, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int, loss::Function)

Select the best machine from a vector based on the lowest Monte Carlo error.
"""
function get_best_machine(machs::Vector, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int, loss::Function)
    isempty(machs) && throw(ArgumentError("Input machine vector cannot be empty."))

    best_error = Inf
    best_mach_idx = 1

    for (idx, mach) in enumerate(machs)
        err = monte_carlo_error(mach, truth_func, domain, N, loss)
        if err < best_error
            best_error = err
            best_mach_idx = idx
        end
    end

    return (machs[best_mach_idx], best_error)
end

"""
    mach_to_func(mach)

Wrap a trained MLJ machine into a simple function `f(x)` that predicts the output for a single scalar input.
"""
function mach_to_func(mach)
    function predictor_func(x_val::Real)
        # Create a DataFrame with the correct feature name used during training
        input_df = DataFrame(x = [x_val])
        # Predict and return the single prediction value
        return MLJ.predict(mach, input_df)[1]
    end
    return predictor_func
end

"""
    get_all_losses(mach, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int)

Calculate standard L1 (Mean Absolute Error) and L2 (Mean Squared Error) for a
machine's predictions against a truth function using Monte Carlo estimation.
"""
function get_all_losses(mach, truth_func::Function, domain::Tuple{Float64,Float64}, N::Int)
    loss_l1(pred, truth) = abs(pred - truth)
    loss_l2(pred, truth) = (pred - truth)^2

    l1 = monte_carlo_error(mach, truth_func, domain, N, loss_l1)
    l2 = monte_carlo_error(mach, truth_func, domain, N, loss_l2)

    return (l1, l2)
end

"""
    train_tuned(df::DataFrame; cv_folds=5, tuning_grid_size=16, max_neighbors=1500)

Train a k-Nearest Neighbor regression model using `MLJ.jl`, performing hyperparameter tuning.
"""
function train_tuned(df::DataFrame; cv_folds=5, tuning_grid_size=16, max_neighbors=1500)
    nrow(df) < 20 && @warn "Training data has very few samples ($(nrow(df))). Model performance may be poor."
    nrow(df) < cv_folds && throw(ArgumentError("Number of data points ($(nrow(df))) is less than number of CV folds ($cv_folds)."))

    y_data = df.y
    x_data = select(df, :x) # Select as DataFrame

    # Define KNN model and tuning ranges
    knn_model = KNNRegressor()
    # Ensure upper K is not more than available points (minus 1 for leave-one-out possibility in CV)
    upper_k = min(max_neighbors, nrow(df) - ceil(Int, nrow(df)/cv_folds), nrow(df)-1)
    lower_k = min(20, upper_k) # Ensure lower <= upper
    (lower_k >= upper_k) && (lower_k = max(1, upper_k - 1)) # Adjust if upper_k is very small

    knn_r_k = range(knn_model, :K; lower=lower_k, upper=upper_k, scale=:log10) # Log scale often better for K
    knn_r_leaf = range(knn_model, :leafsize; lower=50, upper=100) # Linear scale seems reasonable

    # Setup TunedModel
    tuned_knn = TunedModel(
        model=knn_model,
        resampling=CV(nfolds=cv_folds, rng=123), # Set rng for reproducibility
        tuning=Grid(resolution=tuning_grid_size, rng=456), # resolution defines points per range approx. grid size
        ranges=[knn_r_k, knn_r_leaf], # Fix: use `ranges` instead of `range` 
        measure=L2DistLoss(), # Standard Mean Squared Error for regression
        acceleration=CPUThreads(), # Use threading for tuning
        acceleration_resampling=CPUThreads() # Use threading for resampling within tuning
    )

    # Create and fit the machine
    mach = machine(tuned_knn, x_data, y_data)
    @info "Starting KNN hyperparameter tuning..."
    fit!(mach)
    @info "Tuning complete. Best model found."
    rep = report(mach)
    best_params = rep.best_model
    @info "Best KNN hyperparameters:" K=best_params.K leafsize=best_params.leafsize

    return mach
end

# --- Experiment Function ---

"""
    run_estimation_pipeline(sim_params::Dict, est_params::Dict)

Core pipeline: Generate SPDE solution -> Perform Partial Integration -> Train Model -> Evaluate.
"""
function run_estimation_pipeline(sim_params::Dict, est_params::Dict)
    # --- Parameter extraction with defaults ---
    # Simulation
    σ_func = get(sim_params, :sigma_func, x -> 1.0)
    nx = get(sim_params, :nx, 2^9)
    nt = get(sim_params, :nt, 2^10) # Note: Original nt=2^18 is huge, may need adjustment
    L = get(sim_params, :L, 1.0)
    t_buffer = get(sim_params, :t_buffer, 0.1)
    t_end = get(sim_params, :t_end, 0.4)
    u0_val = get(sim_params, :u0_val, 2π)
    solver = get(sim_params, :solver, EM())
    ensemble_size = get(sim_params, :ensemble_size, 1) # Generate multiple trajectories if > 1

    # Estimation
    eps_phys = get(est_params, :eps, 0.01) # Physical size for integration
    max_samples = get(est_params, :max_samples, DEFAULT_MAX_SAMPLES)
    min_samples = get(est_params, :min_samples, MIN_TRAINING_SAMPLES)
    points_per_time = get(est_params, :points_per_time, DEFAULT_POINTS_PER_TIME)
    boundary_margin = get(est_params, :boundary_margin, DEFAULT_BOUNDARY_MARGIN)
    eval_N = get(est_params, :eval_N, 10000)
    eval_q = get(est_params, :eval_domain_quantile, (0.05, 0.95))
    cv_folds = get(est_params, :train_cv_folds, 5)
    tune_grid = get(est_params, :train_tuning_grid, 16)
    max_k = get(est_params, :train_max_neighbors, 1500)

    # Calculate dx, dt based on simulation params
    dx = L / (nx - 1)
    dt_save = (t_end > 0) ? t_end / (nt - 1) : 1e-6 # Used for partial integration dt

    # --- Data Generation ---
    full_df = DataFrame(x=Float64[], y=Float64[])
    trajectories_generated = 0
    
    while nrow(full_df) < min_samples
        trajectories_needed = max(1, ceil(Int, (min_samples - nrow(full_df)) / (points_per_time * nt * 0.5))) # Rough estimate
        actual_ensemble_size = min(ensemble_size, trajectories_needed) # Adapt if ensemble_size is large
        @info "Generating $actual_ensemble_size solution trajectories..."
        solution = generate_solution(σ_func, nx, nt; L, t_buffer, t_end, u0_val, solver, ensemble_size=actual_ensemble_size)
        trajectories_generated += actual_ensemble_size

        # Process each trajectory
        process_trajectory = function(traj_idx)
            # Extract solution matrix: u[space, time]
            sol_matrix = try
                 hcat(solution[traj_idx].u...)
            catch # Handle case where ensemble_size=1
                 hcat(solution.u...)
            end
            @info "Processing trajectory $traj_idx: solution size $(size(sol_matrix))"
            # Ensure Float64 for L_op and integration
            sol_matrix = convert(Matrix{Float64}, sol_matrix)
            partial_integration(sol_matrix, dt_save, dx, eps_phys; max_samples, points_per_time, boundary_margin)
        end

        dfs_partial = [process_trajectory(i) for i in 1:actual_ensemble_size]

        # Combine results
        new_df = vcat(dfs_partial...)
        if nrow(new_df) > 0
             full_df = vcat(full_df, new_df)
             @info "Collected $(nrow(new_df)) samples. Total samples: $(nrow(full_df))."
        else
             @warn "No samples generated from the latest trajectory batch."
             # Avoid infinite loop if partial_integration consistently fails
             if trajectories_generated > ensemble_size * 5 # Arbitrary limit
                 error("Failed to generate sufficient samples after $trajectories_generated trajectories.")
             end
        end
    end

    # Optional: Subsample if we collected vastly more than max_samples
    if nrow(full_df) > max_samples
        @info "Subsampling training data from $(nrow(full_df)) to $max_samples."
        rand_indices = sample(1:nrow(full_df), max_samples, replace=false)
        full_df = full_df[rand_indices, :]
    end

    # --- Model Training ---
    @info "Starting model training with $(nrow(full_df)) samples."
    mach = train_tuned(full_df; cv_folds=cv_folds, tuning_grid_size=tune_grid, max_neighbors=max_k)

    # --- Evaluation ---
    @info "Evaluating trained model..."
    # Define evaluation domain based on quantiles of the collected 'u' values
    q_low, q_high = eval_q
    if nrow(full_df)>1 #need at least 2 points for quantiles
        umin_eval = quantile(full_df.x, q_low)
        umax_eval = quantile(full_df.x, q_high)
        # Ensure domain is valid
        if umin_eval >= umax_eval
           @warn "Quantile-based evaluation domain is invalid ($umin_eval, $umax_eval). Falling back to full range."
           umin_eval = minimum(full_df.x)
           umax_eval = maximum(full_df.x)
        end
    else #fallback if only 1 data point
        umin_eval = minimum(full_df.x)
        umax_eval = maximum(full_df.x)
    end
    eval_domain = (umin_eval, umax_eval)

    truth_sq_func(u) = σ_func(u)^2 # We estimate sigma^2
    l1, l2 = get_all_losses(mach, truth_sq_func, eval_domain, eval_N)
    @info "Evaluation complete" L1_Error=l1 L2_Error=l2 EvaluationDomain=eval_domain

    # --- Return Results ---
    results = Dict(
        :machine => mach,
        :training_data => full_df,
        :l1_error => l1,
        :l2_error => l2,
        :evaluation_domain => eval_domain,
        :simulation_params => sim_params,
        :estimation_params => est_params,
        :best_hyperparameters => report(mach).best_model
    )
    return results
end


"""
    partial_integration(solution::AbstractMatrix{T}, dt::Real, dx::Real, 
                        x_quot::Integer, t_quot::Integer, eps::Real) where T<:Real

Perform partial integration on a solution matrix with an integration window of size `eps`.

# Arguments
- `solution`: Matrix containing the solution data
- `dt`: Time step size
- `dx`: Space step size
- `x_quot`: Quotient for spatial downsampling
- `t_quot`: Quotient for temporal downsampling
- `eps`: Integration window size

# Returns
- `DataFrame`: DataFrame with `:x` (original values) and `:y` (integrated values)
"""
function partial_integration(solution::AbstractMatrix{T}, dt::Real, dx::Real, 
                            x_quot::Integer, t_quot::Integer, eps::Real) where T<:Real
    nx, nt = size(solution)
    
    # Calculate new step sizes
    new_dx = x_quot * dx
    new_dt = t_quot * dt
    
    # Calculate window sizes in grid points
    x_eps = div(eps, dx)  # Integer division (no need for |> Int)
    t_eps = div(eps, dt)
    
    # Apply L operator to the solution
    L_result = L_op(solution, new_dt, new_dx) .* 1/(sqrt(2)*eps)
    
    x_len, t_len = size(L_result)
    
    # Define sampling parameters
    margin = 32  # Safety margin to stay away from boundaries
    
    # Ensure we have enough points for integration
    if x_len <= 2*x_eps + 2*margin || t_len <= t_eps + 10
        error("Solution matrix too small for the requested integration window")
    end
    
    # Define the safe range for x sampling
    x_safe_min = x_eps + margin
    x_safe_max = x_len - x_eps - margin
    
    # Calculate sampling parameters
    max_x_points = x_safe_max - x_safe_min + 1
    num_x_samples = min(10, max_x_points)
    max_total_samples = 50000
    total_desired_samples = min(max_total_samples, t_len * num_x_samples)
    
    # Calculate sampling factor to achieve desired sample count
    sample_factor = ceil(Int, t_len * num_x_samples / total_desired_samples)
    
    # Create channel for results with reasonable buffer size
    results = Channel{Tuple{T, Float64}}(min(1000, total_desired_samples))
    
    # Process in parallel
    Threads.@threads for t in 1:sample_factor:t_len-t_eps-10
        # Sample x points for this timestep
        rand_x = sample(x_safe_min:x_safe_max, num_x_samples, replace=false)
        
        for x in rand_x
            # Extract window for integration
            integrated_view = view(L_result, x-x_eps:x+x_eps, t:t+t_eps)
            
            # Create range objects for integration
            l1, l2 = size(integrated_view)
            rx = range(0, step=dx, length=l1)
            rt = range(0, step=dt, length=l2)
            
            # Compute integration
            integrated = trapz((rx, rt), integrated_view)^2
            
            # Get original value
            u = solution[x, t]
            put!(results, (u, integrated))
        end
    end
    
    close(results)
    
    # Collect results and convert to DataFrame
    collected_results = collect(results)
    return DataFrame(:x => first.(collected_results), :y => last.(collected_results))
end

end # END MODULE SPDE