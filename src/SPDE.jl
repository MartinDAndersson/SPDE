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
#using PyCall
#const KNeighborsRegressor = PyNULL()
#function __init__()
#    @eval @sk_import neighbors: KNeighborsRegressor
#end
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


function alternate_method(u,dt,dx)
    Lu = L_op(u,dt,dx)
end

function integrate(Lu,dx,dt,ϵ)
    nx,nt = size(Lu)
    x_dims,t_dim = size(Lu)
    x_eps = (ϵ ÷ dx) |> Int 
    t_eps = (ϵ ÷ dt) |> Int
    #size(Lu)
    x0 = nx ÷ 2 |> Int
    #@show x0 + x_eps
    t0 = 1
    integration_matrix = Lu[x0-x_eps:x0+x_eps,t0:t0+t_eps]
    integrated = (1/ϵ*dt*dx*sum(integration_matrix[2:end-1,1:end-1]))^2
    return integrated
end

function one_loop(u,dt,dx,ϵ)
    nx,nt = size(u)
    x_eps = (ϵ ÷ dx) |> Int 
    t_eps = (ϵ ÷ dt) |> Int
    t0=1
    x0 = nx ÷ 2 |> Int
    u = u[x0-x_eps-1:x0+x_eps+1,t0:t0+t_eps+1]
    u0 = u[x_eps,t0]
    Lu = L_op(u,dt,dx)
    sample = integrate(Lu,dx,dt,ϵ)
    return u0,sample
end

function whole_thing(ensemble,dt,dx,eps,ϵ)
    t_length= ensemble[1].t |> length
    data = []
    @show x_eps = (eps ÷ dx) |> Int 
    @show t_eps = (eps ÷ dt) |> Int
    @show x_ϵ = (ϵ ÷ dx) |> Int 
    @show t_ϵ = (ϵ ÷ dt) |> Int
    num_x_samples = 20
    total_samples = min(50000,t_length)
    factor = t_length*num_x_samples/total_samples |> x-> ceil(Int,x)
    for sol in ensemble
        Lu = L_op(sol,dt,dx)
        x_length = size(Lu)[1]
        x_mid = x_length ÷ 2 |> Int
        for t in 1:factor:t_length-t_eps
                for x0 in 1:num_x_samples #x_eps+3:10:x_length-x_eps-3
                    x0 = rand(10:x_length-10)
                    #sol_copy = @views sol[:,t:t+t_eps+1]
                    #Lu_copy = @views Lu[:,t:t+t_eps+1]
                    Lu_copy = Lu[x0:x0+x_eps-1,t:t+t_eps-1] #Lu[x0-x_eps:x0+x_eps,t:t+t_eps]
                    u = sol[x0,t]
                    integrated = (1/eps*dt*dx*sum(Lu_copy))^2
                    push!(data,(u,integrated))
                end
        end
    end
    return data
end
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

function get_drift_diffusion() 
	function heat_drift!(du, u, p, t)
		@inbounds begin
		du[1] = 0.0
		du[end] = 0.0
		end
		M = length(du)
		dx,σ = p
		@inbounds for i in 2:M-1
			du[i] = 1/2 .* (u[i-1] - 2u[i] + u[i+1])/(dx^2)
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
    #=
    @show new_nx = nx ÷ x_quot
    @show new_nt = nt ÷ t_quot
    @show new_dx = x_quot*dx
    @show new_dt = t_quot*dt
    @show x_eps = (eps ÷ new_dx) |> Int
    @show t_eps = (eps ÷ new_dt) |> Int
    println("new_nx= $new_nx")
    println("new_nt= $new_nt")
    println("x_eps= $x_eps")
    println("r_eps= $t_eps")
    =#

    #df = DataFrame(:x=>Float64[],:y => Float64[])
    #@show a,b = size(solution)
function partial_integration(solution,dt,dx,L,tmax,x_quot,t_quot,eps)
    nx,nt = size(solution)
    #df = DataFrame(:x=>Float64[],:y => Float64[])
    new_nx = nx ÷ x_quot
    new_nt = nt ÷ t_quot
    new_dx = x_quot*dx
    new_dt = t_quot*dt
    x_eps = (eps ÷ new_dx) |> Int
    t_eps = (eps ÷ new_dt) |> Int
    x_idx = 1:x_quot:nx#[i for i in 1:new_nx]*x_quot
    t_idx = 1:t_quot:nt#[t for t in 1:new_nt]*t_quot
    new_sol = solution[x_idx,t_idx]#downsample_matrix(solution,x_quot,t_quot)
    new_dx = x_quot*dx
    new_dt = t_quot*dt
    Lu_clean = L_op(new_sol,new_dt,new_dx) #.* sqrt(dx*dt) 
    Lu = Lu_clean
    #Lu = Lu * 1/sqrt(eps) * new_dx # no time
    Lu .= Lu * 1/eps * new_dx * new_dt # with time
    buffer = 1#2^13 ÷ t_quot
    x_len,t_len = size(Lu_clean)
    #time_startup = 2^15 ÷ t_quot
    max_x_points = (x_len)-x_eps-1 -(x_eps+1)
    num_x_samples = min(20,max_x_points)
    total_samples = min(50000,t_len*num_x_samples)
    factor = t_len*num_x_samples/total_samples |> x-> ceil(Int,x)
    results = Channel{Tuple}(Inf)
    Threads.@threads for t in 1:factor:t_len-t_eps-10
        rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
        for i in 1:num_x_samples
            #x = rand(x_eps+1:x_len-x_eps-1)
            x=rand_x[i]
            integrated_view = view(Lu, x:x+x_eps+1, t:t+t_eps+1) # right now not shifted to -1 to compensate
            l1,l2 = size(integrated_view)
            integrated=trapz((1:l1,1:l2),integrated_view)^2
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




function experiment_2(solution,dt,dx,L,tmax,x_quot,t_quot,eps,sigma)
	df = partial_integration(solution,dt,dx,L,tmax,x_quot,t_quot,eps)
	@show x_size = size(df,1)
	if x_size < 50
		return nothing
	end
    df_train,df_test = partition(df,0.8,rng=123)
    σ = sigma[1]
    sigma_name = sigma[2]
	new_dx = x_quot*dx
	new_dt = t_quot*dt
	#df=DataFrame(Dict(:x=>first.(data),:y=>last.(data)))
	x_data = select(df_train,1)
	y_data = df_train.y #.* nx/t_quot
	#@show size(df)
	max_nbors = min(300,length(df.x))
    knn=KNN()
    r=range(knn,:K,lower=20,upper=max_nbors)
    knn_tuned = TunedModel(model=knn,resampling=CV(nfolds=3),
    tuning=Grid(resolution=20), range=r,measure=l2)
	#gridsearch=GridSearchCV(KNeighborsRegressor(),Dict(:n_neighbors=>20:20:max_nbors));
	#ScikitLearn.fit!(gridsearch,x_data[:,:],y_data);
    mach = machine(knn_tuned,x_data,y_data)
    fit!(mach)
	if size(x_data,1) < 15000
		x = x_data
	else
        samples_idxs = sample(1:size(df_train, 1), 10000, replace=false)
		x=x_data[samples_idxs,:] 
	end
    x=select(df_test,1)
	#p1=scatter(df.x,df.y,alpha=0.1,label=L"(L^hu)^2")
	#plot!(x,x->σ(x)^2,label=L"\sigma^2",linewidth=3.,title="estimates")
	#plot!(x,predict(gridsearch,x[:,:]),label="K-nearest neighbor",linewidth=3.)
	#lims!(0,3)
	#p2=scatter(σ.(x).^2 ./ predict(gridsearch,x[:,:]),label="quotient")
	#scatter!(df.x,df.y,alpha=0.1)

	l1_loss = abs.(σ.(x.x).^2 .- predict(mach,x)) |> mean
	l1_std = abs.(σ.(x.x).^2 .- predict(mach,x)) |> std
	l2_loss = abs.(σ.(x.x).^2 .- predict(mach,x)).^2 |> mean
	l2_std = abs.(σ.(x.x).^2 .- predict(mach,x)).^2 |> std
	quotient = (σ.(x.x).^2 ./ predict(mach,x)) |> x->filter(!isnan,x) |> mean
	return (1,(eps,new_dx,new_dt,l1_loss,l1_std,l2_loss,l2_std,quotient,x_size,sigma_name))
end

function experiment_3(u_begin,nx,samples,dt,dx,L,tmax,algo,t_idxs,sigmas)
	df_loss = DataFrame([:eps=>Float64[],:new_dx=>Float64[],:new_dt=>Float64[],:l1_loss=>Float64[],:l1_std=>Float64[],:l2_loss=>Float64[],:l2_std=>Float64[],:quotient=>Float64[],:x_size=>Float64[],:sigma=>String[]])
	drift,diff = get_drift_diffusion()
    jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
	SDE_sparse = SDEFunction(drift,diff;jac_prototype=jac_prot)
    t0=tmax;
    ϵ = 32dx
    time_range = ((0.0,t0+ϵ+dt))
	#σ = sigma
	#p=(dx,σ)
	#E=EnsembleProblem(SDEProblem(SDE_sparse,u_begin,time_range,p))
	for i in 1:samples, sigma in sigmas
		#E=E_sigma[1]
		#sigma = E_sigma[2]
		σ = sigma[1]
		sigma_name = sigma[2]
		p=(dx,σ)
		E=EnsembleProblem(SDEProblem(SDE_sparse,u_begin,time_range,p))
		solution=solve(E,dtmax=dt,trajectories=1,saveat=t_idxs,progress=true,algo,maxiters=1e7)[1]
        println(solution.t[end])
		for j in 0:4#0:4
			res=[experiment_2(solution,dt,dx,L,tmax,2^i,2^i*3,(2^(i+j))*dx,sigma) for i in 7:-1:0] #7:-1:0
			filter!(!isnothing,res)
			push!(df_loss,last.(res[:])...)
		end
	end
	return df_loss
end

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

function mach_to_func(mach)
	est_wrapper(x) = MLJ.predict(mach,DataFrame(:x=>[x]))[1]
	return est_wrapper
end

function get_all_losses(mach,truth,domain,N)
	losses = Dict("l1"=>x->abs(x),"l2"=>x->abs(x)^2)
	l1 = monte_carlo_error(mach,truth,domain,N,losses["l1"])
	l2 = monte_carlo_error(mach,truth,domain,N,losses["l2"])
	return (l1,l2)
end


function main_exp(spde_params)
    sigma_1(x) =  1.
    sigma_2(x) = 0.5*sin(2*6*pi/2*x)#0.5x
    sigma_3(x) = 0.5*sin(6*x/(2*pi))
    sigma_4(x) = 0.5x
    sigmas = Dict("sigma1"=>sigma_1,"sigma2"=>sigma_2,"sigma3"=>sigma_3,"sigma4"=>sigma_4)
    @unpack xquot, tquot, epsilon, sigma = spde_params
    σ = sigmas[sigma]  #sigmas[sigma]
    L=1
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
    solution = generate_solution(σ)
    println(size(solution))
    df = DataFrame(:x=>Float64[],:y=>Float64[])
    while size(df,1) < 20000
        solution = generate_solution(σ)
        df_partial=SPDE.partial_integration(Matrix(solution),dt,dx,L,tmax,xquot,tquot,epsilon)
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

function generate_solution(σ)
    algo = ImplicitEM(linsolve = KLUFactorization())
    L=1; 
	tmax = 0.3;
	h=2^9
	nx = h;
	nt= 2^18
	dx = L/(nx-1); 
	dt = tmax/(nt-1); 
    max_ϵ = 32dx
	u_begin = 2*pi*ones(nx); u_begin[1] = 0; u_begin[end] = 0;
	drift,diff = SPDE.get_drift_diffusion()
    t0=tmax;
	jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
	SDE_sparse = SPDE.SDEFunction(drift,diff;jac_prototype=jac_prot)
	#t0=tmax;
	#ϵ = 32dx
	time_range = ((0.0,t0+max_ϵ+dt)) # fixa detta
    t_idxs = 2^14*dt:dt:tmax
    p=(dx,σ)
    E=EnsembleProblem(SDEProblem(SDE_sparse,u_begin,time_range,p))
    solution=solve(E,dtmax=dt,trajectories=1,saveat=t_idxs,progress=true,algo,maxiters=1e7)[1]
    return solution
end

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
	tuning=Grid(goal=6),
	range=[knn_r1,knn_r2],measure=rms,
	acceleration=CPUThreads(),
	acceleration_resampling=CPUThreads());
    mach = machine(knn_tuned,x_data,y_data)
    fit!(mach)
    return mach
end

end
#=
function whole_thing_lamperti(sol,dt,dx,eps,ϵ)
    x_length,t_length = size(sol)
    data = []
    @show x_eps = (eps ÷ dx) |> Int 
    @show t_eps = (eps ÷ dt) |> Int
    @show x_ϵ = (ϵ ÷ dx) |> Int 
    @show t_ϵ = (ϵ ÷ dt) |> Int
    #sol = ψ_inv.(Array(sol))
    Lu = L_op(sol,dt,dx)
    x_length = size(Lu)[1]
    @show x0 = x_length ÷ 2 |> Int
    for t in 100:1000:t_length-t_eps-1
            for x0 in [x0]#x_eps+3:1:x_length-x_eps-3	
            #sol_copy = @views sol[:,t:t+t_eps+1]
            #Lu_copy = @views Lu[:,t:t+t_eps+1]
            Lu_copy = Lu[x0-x_eps:x0+x_eps,t:t+t_eps] #Lu[x0-x_eps:x0+x_eps,t:t+t_eps]
            u = sol[x0,t]
            integrated = (1/eps*dt*dx*sum(Lu_copy))^2
            push!(data,(u,integrated))
            end
    end
    return data
end
=#
 # module Volatility