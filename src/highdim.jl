"""
    highdim

Module for handling higher-dimensional stochastic partial differential equations.

This module provides tools for generating, analyzing, and estimating diffusion
coefficients in 2D and higher dimensional SPDEs. It includes specialized versions
of the core algorithms that work with multi-dimensional data.

Key functionality includes:
- Multi-dimensional L-operator implementation
- SPDE solution generation for 2D problems
- Partial integration methods for higher dimensions
- GPU-accelerated computation for large systems
"""
module highdim
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
using Symbolics
using CUDA
KNN = MLJ.@load KNNRegressor
using Trapz
export f
f(x) = 3

"""
    L_op(u, dt, dx)

Apply the differential operator to a 3D solution tensor to extract information 
about the diffusion coefficient in a 2D SPDE.

This operator implements: L[u] = ∂u/∂t - (1/2)∇²u 
where ∇² is the 2D Laplacian operator (∂²/∂x² + ∂²/∂y²)

# Arguments
- `u::Array{Float64,3}`: Solution tensor with dimensions (x, y, time)
- `dt::Float64`: Time step size
- `dx::Float64`: Space step size (assumed equal in x and y)

# Returns
- `Lu::Array{Float64,3}`: Tensor with the operator applied at each valid point

This function uses first-order finite differences for time derivatives and 
second-order central differences for spatial derivatives in both x and y dimensions.
The implementation is multi-threaded for performance with large arrays.
"""
function L_op(u,dt,dx)
    Lu = similar(u)
    nnx,nnxy,nnt = size(u)
    Threads.@threads for t in 1:nnt-1
        @inbounds for x in 2:nnx-1
            @inbounds for y in 2:nnx-1
                t_diff = 1/dt*(u[x,y,t+1]-u[x,y,t])
                x_diff = (1/dx^2)*(u[x-1,y,t]-2*u[x,y,t]+u[x+1,y,t])
                y_diff = (1/dx^2)*(u[x,y-1,t]-2*u[x,y,t]+u[x,y+1,t])
                xy_diff = x_diff + y_diff
                Lu[x,y,t] = t_diff - 1/2 * xy_diff
            end
        end
    end
    return Lu
end
"""
    generate_solution(σ, h, nt)

Generate numerical solutions to a 2D stochastic heat equation.

# Arguments
- `σ::Function`: Diffusion coefficient function
- `h::Int`: Number of spatial grid points in each dimension (h×h grid)
- `nt::Int`: Number of time steps

# Returns
- `solution::RODESolution`: Solution from the SDE solver

This function solves the 2D stochastic heat equation:
∂u/∂t = (1/2)∇²u + σ(u)∂W/∂t

where ∇² is the 2D Laplacian. It uses the SROCK1 algorithm, which provides
good stability properties for stiff stochastic systems.

The implementation uses Symbolics.jl to automatically generate the Jacobian
sparsity pattern, which improves solver performance.
"""
function generate_solution(σ,h,nt)
    #algo = ImplicitRKMil(linsolve = KLUFactorization())#(linsolve = KLUFactorization())
    algo = SROCK1()
    #algo=ImplicitEM(linsolve = KLUFactorization())
    L=1; 
	#tmax = 0.3;
    tmax=0.4
    t0=0.
	#h=2^6
    N=h
	nx = h;
	nt= nt
	dx = L/(nx-1); 
	dt = tmax/(nt-1); 
    max_ϵ = 32dx
	u_begin = 12*ones(nx,nx)
	u_begin[1,:] .= 0; u_begin[end,:] .= 0; u_begin[:,1] .= 0; u_begin[:,end] .=0
	drift,diff = drift!,noise!
    p=(dx,σ,N)
    u0 = rand(N,N)
	du0 = copy(u0)
    jac_sparsity = Symbolics.jacobian_sparsity((du,u) -> drift!(du,u,p,0.0),du0,u0)
	#jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
	SDE_sparse = SDEFunction(drift,diff;jac_prototype=float.(jac_sparsity))
	#t0=tmax;
	#ϵ = 32dx
    xmid = nx ÷ 2
    eps = 4
    x_idxs= CartesianIndices((xmid-eps:xmid+eps,xmid-eps:xmid+eps))
	@show time_range = (t0,tmax+dt) # fixa detta
    t_idxs = 0.1:2*dt:0.3
    E=SDEProblem(SDE_sparse,u_begin,time_range,p)
    solution=solve(E,dtmax=dt,progress=true,algo,maxiters=1e7,saveat=t_idxs,dt=dt)
    return solution
end

function test()
    sigma_1(x) =  0.5
    sigma_2(x) = 0.5*sin(2*6*pi/2*x)#0.5x
    sigma_3(x) = 0.5*sin(6*x/(2*pi))
    sigma_4(x) = 0.1*sin(x)
    xquot=2
    tquot=2
    #sigmas = Dict("sigma1"=>sigma_1,"sigma2"=>sigma_2,"sigma3"=>sigma_3,"sigma4"=>sigma_4)
    #@unpack xquot, tquot, epsilon, sigma = spde_params
    σ = sigma_1 #sigmas[sigma]  #sigmas[sigma]
    L=1
    tmax = 0.3;
	#σ = x-> 0.5*sin(2pi*x/6)
    truth(x) = σ(x)^2
    h=2^6
    nx = h;
	nt= 2^15
    dx = L/(nx-1); 
	dt = tmax/(nt-1);
    epsilon=2dx
    tmax = 0.3; 
   #solution = generate_solution(σ,h,nt)
    df = DataFrame(:x=>Float64[],:y=>Float64[])
    while size(df,1) < 20000
        solution = generate_solution(σ,h,nt)
        df_partial=partial_integration(solution,dt,dx,xquot,tquot,epsilon)
        df = vcat(df,df_partial)
    end
    max_size_df = min(50000,size(df,1))
    rand_rows = sample(1:size(df,1),max_size_df,replace=false)
    df=df[rand_rows,:]
    println(size(df,1))
    mach = train_tuned(df)
    N=10000
    domain=(2,6)
    #l1,l2 = get_all_losses(mach,truth,domain,N)
    #fulld = copy(spde_params)
    #fulld["l1"] = l1
    #fulld["l2"] = l2
    return mach,df,truth
end

"""
    partial_integration(solution, dt, dx, x_quot, t_quot, eps)

Perform the partial integration method for 2D SPDEs to extract diffusion coefficient information.

# Arguments
- `solution::Array{Float64,3}`: 3D solution tensor with dimensions (x, y, time)
- `dt::Float64`: Time step size of the original solution
- `dx::Float64`: Space step size of the original solution (assumed equal in x and y)
- `x_quot::Int`: Spatial downsampling factor (applied to both x and y dimensions)
- `t_quot::Int`: Temporal downsampling factor
- `eps::Float64`: Size of the local integration window (in original dx units)

# Returns
- `df::DataFrame`: DataFrame with columns :x (u values) and :y (corresponding estimated σ²(u) values)

This is the 2D version of the partial integration algorithm, extending the 1D method to
handle 2D spatial domains. It uses multi-dimensional numerical integration via the Trapz
package and parallelizes computation with Julia's threading.
"""
function partial_integration(solution,dt,dx,x_quot,t_quot,eps)
	#eps = 4*dx
    #solution = Array(solution)
	nx,ny,nt=size(solution)
	new_dx = x_quot*dx
	new_dt = t_quot*dt
	x_eps = (eps ÷ new_dx) |> Int
	t_eps = (eps ÷ new_dt) |> Int
    y_eps = x_eps
	x_idx = 1:x_quot:nx#[i for i in 1:new_nx]*x_quot
    t_idx = 1:t_quot:nt
	y_idx = 1:x_quot:nx#[t for t in 1:new_nt]*t_quot
    new_sol = @view solution[x_idx,y_idx,t_idx]#downsample_matrix(solution,x_quot,t_quot)
    new_dx = x_quot*dx
    new_dt = t_quot*dt  
    Lu=L_op(new_sol,new_dt,new_dx) .* 1/eps .* new_dx^2 .* new_dt
    x_len,y_len,t_len = size(Lu)
    	    
	    #time_startup = 2^15 ÷ t_quot
    max_x_points = (x_len)-x_eps-1 -(x_eps+1)
    num_x_samples = min(20,max_x_points-6) # 20 usually
    total_samples = min(50000,t_len*num_x_samples)
    factor = t_len*num_x_samples/total_samples |> x-> ceil(Int,x)
    results = Channel{Tuple}(Inf)
	Threads.@threads for t in 1:factor:t_len-t_eps-10
        #plotln(t)
		rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		rand_y = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		for i in 1:num_x_samples
			x=rand_x[i]
			y=rand_y[i]
			integrated_view = view(Lu, x:x+x_eps,y:y+y_eps, t:t+t_eps)
			l1,l2,l3 = size(integrated_view)
			integrated=trapz((1:l1,1:l2,1:l3),integrated_view)^2
			u=new_sol[x,y,t]
			put!(results, (u, integrated))
		end
	end
	close(results)
	collected_results = collect(results)
    df = DataFrame(:x=>first.(collected_results),:y=>last.(collected_results))
    return df
end

"""
    partial_new(sol, dt, dx, eps)

Alternative implementation of the partial integration method using fixed spatial locations.

# Arguments
- `sol::Array{Float64,3}`: 3D solution tensor
- `dt::Float64`: Time step size
- `dx::Float64`: Space step size
- `eps::Int`: Size of integration window in grid points

# Returns
- `df::DataFrame`: DataFrame with (u, σ²(u)) data points

This function is an experimental alternative to the main partial_integration method,
using a fixed spatial location (at 1/4 of the domain) rather than random sampling.
It provides a simpler implementation for testing and comparison.
"""
function partial_new(sol,dt,dx,eps)
    nx,ny,nt = size(sol)
    #Lu = similar(sol)
    @show t_eps = (dx ÷ dt)*eps |> Int
    sol_array = @view sol[:,:,:]
    Lu=L_op(sol_array,dt,dx)
    df = DataFrame(:x=>Float64[],:y=>Float64[])
    #c = dt*dx^2
    mid = nx ÷ 4
    @inbounds for t in 2:nt-1
        x_idxs,y_idxs = (mid,mid)
        integrate_view = @view Lu[x_idxs:x_idxs+eps,y_idxs:y_idxs+eps,t:t+t_eps]
        l1,l2,l3 = size(integrate_view)
        x_range=dx:dx:l1*dx 
        y_range=dx:dx:l2*dx 
        t_range=dt:dt:l3*dt
        #t_range=(0,l3*dt)
        if eps > 0
            integ = trapz((x_range,y_range,t_range),integrate_view) * (1/dx)^2
        else
            integ = integrate_view[1] * dt
        end
        integ = integ^2
        u=sol[x_idxs,y_idxs,t]
        push!(df,(u,integ))
    end
    return df 
end

"""
    drift!(du, u, p, t)

Implements the deterministic part (Laplacian) of the 2D stochastic heat equation.

# Arguments
- `du::Array{Float64,2}`: Output array for the drift term
- `u::Array{Float64,2}`: Current solution state
- `p::Tuple`: Parameters (dx, σ, N) where dx is the grid spacing, σ is the diffusion function, 
              and N is the grid size
- `t::Float64`: Current time

This function computes ∇²u, the 2D Laplacian of u, using finite differences.
It sets zero Dirichlet boundary conditions at the domain boundaries.
"""
function drift!(du,u,p,t)
    dx,σ = p
    du[1,:] .= 0
    du[end,:] .= 0
    du[:,1] .= 0
    du[:,end] .= 0
    N=p[3]
    @inbounds for i in 2:N-1
        @inbounds for j in 2:N-1
            dux = 0.5 .* (u[i-1,j]-2u[i,j] + u[i+1,j])
            duy = 0.5 .* (u[i,j-1]-2u[i,j] + u[i,j+1])
            du[i,j] = (dux + duy)/(dx^2)
        end
    end
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
    noise!(du, u, p, t)

Implements the stochastic part of the 2D stochastic heat equation.

# Arguments
- `du::Array{Float64,2}`: Output array for the noise term
- `u::Array{Float64,2}`: Current solution state
- `p::Tuple`: Parameters (dx, σ, N) where dx is the grid spacing, σ is the diffusion function, 
              and N is the grid size
- `t::Float64`: Current time

This function computes σ(u) * sqrt(1/dx²), implementing the state-dependent 
diffusion coefficient for the noise term. It uses broadcasting for efficient
vectorized computation.
"""
function noise!(du,u,p,t)
    dx,σ,N = p
    #@inbounds for i in 2:N-1
    #    @inbounds for j in 2:N-1
    #        du[i,j] = σ(u[i,j])*sqrt(1/dx^2)
    #   end
    #end
    du .= σ.(u) .* sqrt(1/dx^2)
end

"""
    train_tuned(df)

Train a k-nearest neighbor regression model with hyperparameter tuning for 2D SPDEs.

# Arguments
- `df::DataFrame`: DataFrame with columns :x (feature values) and :y (target values)

# Returns
- `mach::Machine`: Trained MLJ machine with optimized hyperparameters

This function creates a tuned k-nearest neighbors model with cross-validation,
optimizing the number of neighbors and leaf size for best performance.
It filters very large values (>15) that may represent outliers or boundary effects.

The model is trained to predict the squared diffusion coefficient σ²(u)
from solution values u in the context of 2D SPDEs.
"""
function train_tuned(df)
	df_filtered = filter(row -> row.x < 15, df)
	#df_train,df_test = partition(df_filtered  ,0.8,rng=123)
	x_data = select(df_filtered,1)
	y_data = df_filtered.y #.* nx/t_quot
    max_nbors = min(1500,length(df.x))
	knn=KNN()
	knn_r1=range(knn,:K,lower=20,upper=max_nbors)
	knn_r2=range(knn,:leafsize,lower=50,upper=100)
	knn_r3 = range(knn,:weights,values=[NearestNeighborModels.Dudani(),NearestNeighborModels.Uniform(),
	NearestNeighborModels.DualD(),NearestNeighborModels.DualU()])
	knn_tuned = TunedModel(model=knn,
	resampling=CV(nfolds=5),
	tuning=Grid(goal=12),
	range=[knn_r1,knn_r2],measure=rms,
	acceleration=CPUThreads(),
	acceleration_resampling=CPUThreads());
    mach = machine(knn_tuned,x_data,y_data)
    fit!(mach)
    return mach
end

"""
    gpu_generation(σ, h, nt)

Generate solutions to 2D SPDEs with optimized GPU computation.

# Arguments
- `σ::Function`: Diffusion coefficient function
- `h::Int`: Number of spatial grid points in each dimension (h×h grid)
- `nt::Int`: Number of time steps

# Returns
- `solution::RODESolution`: Solution from the SDE solver

This function is similar to generate_solution, but with settings optimized
for GPU computation with large 2D systems. It uses the SROCK1 algorithm
with automatic Jacobian sparsity detection via Symbolics.jl.

This is typically used for larger spatial domains where GPU acceleration
provides significant performance benefits.
"""
function gpu_generation(σ,h,nt)
    algo = SROCK1()#ImplicitRKMil(linsolve = KLUFactorization())
    L=1; 
	tmax = 1.;
	#h=2^6
    N=h
	nx = h;
	nt= nt
	dx = L/(nx-1); 
	dt = tmax/(nt-1); 
    max_ϵ = 32dx
	u_begin = 6*ones(nx,nx)
	u_begin[1,:] .= 0; u_begin[end,:] .= 0; u_begin[:,1] .= 0; u_begin[:,end] .=0
	drift,diff = drift!,noise!
    t0=tmax;
    p=(dx,σ,N)
    u0 = rand(N,N)
	du0 = copy(u0)
    jac_sparsity = Symbolics.jacobian_sparsity((du,u) -> drift!(du,u,p,0.0),du0,u0)
    #u_begin = cu(u_begin)
	SDE_sparse = SDEFunction(drift,diff;jac_prototype=float.(jac_sparsity))
	time_range = (0.0,tmax+dt) # fixa detta
    t_idxs = 0.3:dt:0.4
    E=SDEProblem(SDE_sparse,u_begin,time_range,p)
    solution=solve(E,dtmax=dt,saveat=t_idxs,progress=true,algo,maxiters=1e7,dt=dt)
    return solution
end


end