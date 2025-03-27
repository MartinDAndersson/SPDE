using LinearAlgebra, Random, Distributions, DifferentialEquations
using SparseArrays, Statistics, MLJ, Trapz, DataFrames
using NearestNeighborModels
Random.seed!(123)

# Load KNN regressor
KNN = @load KNNRegressor

"""
Define drift and diffusion terms for the SDE solver
"""
function get_drift_diffusion(σ) 
    function heat_drift!(du, u, p, t)
        dx = p
        du[1] = 0.0
        du[end] = 0.0
        
        for i in 2:length(u)-1
            du[i] = 0.5 * (u[i-1] - 2*u[i] + u[i+1]) / (dx^2)
        end
    end
    
    function heat_diffusion_noise!(du, u, p, t)
        dx = p
        du[1] = 0
        du[end] = 0
        
        for i in 2:length(u)-1
            du[i] = σ(u[i]) * sqrt(1/dx)
        end
    end
    
    return heat_drift!, heat_diffusion_noise!
end

"""
Apply the L operator to the solution
"""
function compute_L_op(u, dt, dx)
    Lu = similar(u)
    nx, nt = size(u)
    
    for t in 1:nt-1
        for x in 2:nx-1
            time_derivative = (u[x,t+1] - u[x,t]) / dt
            space_derivative = (u[x-1,t] - 2*u[x,t] + u[x+1,t]) / (dx^2)
            Lu[x,t] = time_derivative - 0.5 * space_derivative
        end
    end
    
    # Set boundary values to zero
    Lu[1, :] .= 0
    Lu[end, :] .= 0
    Lu[:, end] .= 0
    
    return Lu
end

"""
Generate solution to the SPDE
"""
function generate_solution(σ, nx, nt)
    # Parameters
    L = 1.0  # Domain length
    tmax = 0.3  # Maximum time
    
    # Discretization
    dx = L/(nx-1)
    dt = tmax/(nt-1)
    
    # Initial condition
    u_begin = 2*pi*ones(nx)
    u_begin[1] = 0
    u_begin[end] = 0
    
    # Setup SDE problem
    drift, diffusion = get_drift_diffusion(σ)
    jac_proto = Tridiagonal([1.0 for _ in 1:nx-1], [-2.0 for _ in 1:nx], [1.0 for _ in 1:nx-1]) |> sparse
    sde_func = SDEFunction(drift, diffusion; jac_prototype=jac_proto)
    
    prob = SDEProblem(sde_func, u_begin, (0.0, tmax), dx)
    
    # Solve the SDE
    dt_save = tmax/100  # Save 100 timepoints
    solution = solve(prob, SROCK1(), dt=dt/10, saveat=dt_save, progress=true)
    
    # Convert solution to matrix
    u_matrix = zeros(nx, length(solution.t))
    for (i, t) in enumerate(solution.t)
        u_matrix[:, i] = solution(t)
    end
    
    return u_matrix, dx, dt_save
end

"""
Perform the estimation at various points using the integrated operator approach
"""
function estimate_sigma_squared(u_matrix, dt, dx, epsilon)
    nx, nt = size(u_matrix)
    
    # Compute L operator
    Lu = compute_L_op(u_matrix, dt, dx)
    
    # Number of points for the integration regions
    eps_points_x = max(1, round(Int, epsilon/dx))
    eps_points_t = max(1, round(Int, epsilon/dt))
    
    # Normalization factor based on theory
    norm_factor = sqrt(2 * epsilon^2)  # For 1D case
    
    # Collect (u, estimated_sigma²) pairs
    results = []
    
    # Skip boundary regions
    x_margin = eps_points_x + 2
    t_margin = eps_points_t + 2
    
    for t in t_margin:2:nt-t_margin
        for x in x_margin:2:nx-x_margin
            # Get the value of u at this point
            u_val = u_matrix[x, t]
            
            # Define integration region (centered at x,t)
            x_range = (x-eps_points_x):(x+eps_points_x)
            t_range = t:(t+eps_points_t)
            
            # Create space-time region for integration
            Lu_region = Lu[x_range, t_range]
            
            # Convert indices to actual coordinates for proper integration
            x_coords = ((x_range .- x) .* dx) .+ 0  # Center at 0
            t_coords = ((t_range .- t) .* dt) .+ 0  # Center at 0
            
            # Perform the integration
            integral = trapz((x_coords, t_coords), Lu_region)
            
            # Square and scale according to theory
            sigma_squared_estimate = (integral / norm_factor)^2
            
            push!(results, (u_val, sigma_squared_estimate))
        end
    end
    
    # Convert to DataFrame
    df = DataFrame(x=first.(results), y=last.(results))
    
    return df
end

"""
Train a nonparametric estimator for σ²
"""
function train_estimator(df)
    # Prepare data
    x_data = df.x
    y_data = df.y
    
    # Setup KNN model with cross-validation
    knn = KNN()
    knn_model = TunedModel(
        model=knn,
        resampling=CV(nfolds=5),
        tuning=Grid(resolution=10),
        range=[range(knn, :K, lower=5, upper=50)],
        measure=LPLoss(p=2),
    )
    
    # Train the model
    mach = machine(knn_model, reshape(x_data, :, 1), y_data)
    fit!(mach)
    
    # Create a function that takes scalar input
    function sigma_squared_estimator(x)
        return MLJ.predict(mach, reshape([x], :, 1))[1]
    end
    
    return sigma_squared_estimator, mach
end

"""
Main function to run the estimation
"""
function main()
    # True diffusion function to recover
    σ_true(x) = 0.5 * x  # Example: linear function
    σ_squared_true(x) = σ_true(x)^2
    
    # Parameters
    nx = 2^8  # Number of spatial points
    nt = 2^16  # Number of time points
    
    # Generate solution
    println("Generating SPDE solution...")
    u_matrix, dx, dt = generate_solution(σ_true, nx, nt)
    
    # Test different epsilon values
    epsilons = [4*dx, 8*dx, 16*dx]
    
    for epsilon in epsilons
        println("Estimating with epsilon = $epsilon...")
        
        # Estimate sigma squared at various points
        df = estimate_sigma_squared(u_matrix, dt, dx, epsilon)
        
        # Train nonparametric estimator
        println("Training nonparametric estimator...")
        sigma_squared_estimator, mach = train_estimator(df)
        
        # Evaluate estimator
        test_points = range(0.1, stop=3.0, length=20)
        estimated = [sigma_squared_estimator(x) for x in test_points]
        true_values = [σ_squared_true(x) for x in test_points]
        
        # Compute error metrics
        l1_error = mean(abs.(estimated .- true_values))
        l2_error = sqrt(mean((estimated .- true_values).^2))
        
        println("L1 error: $l1_error")
        println("L2 error: $l2_error")
        
        # One could plot the results here if using a plotting library
        println("True σ² vs Estimated σ² at test points:")
        for (x, true_val, est_val) in zip(test_points, true_values, estimated)
            println("x = $x, true = $true_val, est = $est_val")
        end
        println()
    end
end

# Run the estimation
main()