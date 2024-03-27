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
using Trapz
using Symbolics
using CUDA
KNN = MLJ.@load KNNRegressor
using Trapz
export f
f(x) = 3

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
#2^27
function generate_solution(σ,h,nt)
    #algo = ImplicitRKMil(linsolve = KLUFactorization())#(linsolve = KLUFactorization())
    algo = SRIW1()
    L=1; 
	tmax = 0.3;
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
	#jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
	SDE_sparse = SDEFunction(drift,diff;jac_prototype=float.(jac_sparsity))
	#t0=tmax;
	#ϵ = 32dx
	time_range = (0.0,tmax+dt) # fixa detta
    t_idxs = 2^11*dt:dt:0.4
    E=SDEProblem(SDE_sparse,u_begin,time_range,p)
    solution=solve(E,dtmax=dt,progress=true,algo,maxiters=1e7,save_everystep = false)
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
    num_x_samples = min(20,max_x_points)
    total_samples = min(50000,t_len*num_x_samples)
    factor = t_len*num_x_samples/total_samples |> x-> ceil(Int,x)
    results = Channel{Tuple}(Inf)
	Threads.@threads for t in 1:factor:t_len-t_eps-10
		rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		rand_y = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		for i in 1:num_x_samples
			x=rand_x[i]
			y=rand_y[i]
			integrated_view = view(Lu, x:x+x_eps+1,y:y+y_eps, t:t+t_eps)
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


function mach_to_func(mach)
	est_wrapper(x) = MLJ.predict(mach,DataFrame(:x=>[x]))[1]
	return est_wrapper
end
# ╔═╡ dc5c0a94-38ac-4f86-ab6e-626eaed699fc
function noise!(du,u,p,t)
    dx,σ,N = p
    #@inbounds for i in 2:N-1
    #    @inbounds for j in 2:N-1
    #        du[i,j] = σ(u[i,j])*sqrt(1/dx^2)
    #   end
    #end
    du .= σ.(u) .* sqrt(1/dx)
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

function gpu_generation(σ,h,nt)
    algo = SRIW1()#ImplicitRKMil(linsolve = KLUFactorization())
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
    u_begin = cu(u_begin)
	SDE_sparse = SDEFunction(drift,diff;jac_prototype=float.(jac_sparsity))
	time_range = (0.0,tmax+dt) # fixa detta
    t_idxs = 2^11*dt:dt:0.4
    E=SDEProblem(SDE_sparse,u_begin,time_range,p)
    solution=solve(E,dtmax=dt,saveat=t_idxs,progress=true,algo,maxiters=1e7)
    return E
end


end