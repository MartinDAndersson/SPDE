### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f612d560-9988-4d3b-a605-8d145d7f434e
using DrWatson

# ╔═╡ 6c541e1a-dd2f-11ee-1948-03057f232327
begin
	@quickactivate "SPDE"
	import Pkg
	using Revise
	#Pkg.activate(".")
	using LinearAlgebra, Random, DifferentialEquations, LsqFit
	using SparseArrays
	using LaTeXStrings, LoopVectorization, DataFrames,CSV, FileIO
	using StatsBase
	import StatsBase,ProgressLogging
	using DataFrames
	using Integrals
	import Plots
	using NearestNeighborModels
	using SPDE
	using MLJ
end

# ╔═╡ a0576e78-e142-4fb6-9b96-c9edc91d1707
using AlgebraOfGraphics, CairoMakie

# ╔═╡ 5a72d3dc-8d13-4aaa-80c9-e0ab9cf35638
using Distributions

# ╔═╡ 6d4016e8-f799-4fea-b0a5-dab23f467848
using Interpolations

# ╔═╡ fe164335-c328-45c7-af42-8d05c515639a
using PlutoUI

# ╔═╡ 58163253-2c77-450e-8a08-901a25159aa7
using Glob

# ╔═╡ 25aa9975-b444-4c2d-9121-7f34e01c577d
using Trapz

# ╔═╡ 6a92067c-958c-48a6-9e64-b94edfb3bb45


# ╔═╡ 85463824-794f-4f9c-85b6-31ecc2f02b06
begin
	L=1; 
	tmax = 0.3;
	h=2^6
	nx = h;
	nt= 2^16
	dx = L/(nx-1); 
	dt = tmax/(nt-1); 
	dt_int=dt
	r=dt/dx^2;
	σ = x-> 0.5x
	#figname = "0.5x"
	ϵ = 32*dx
	samples = 1;
	algo = ImplicitEM(linsolve = KLUFactorization())#SROCKEM(strong_order_1=false)#ImplicitEM(linsolve = KLUFactorization())#ImplicitRKMil(linsolve = KLUFactorization()) #SImplicitMidpoint
	t0=tmax;
	u_begin = 2*pi*ones(nx); u_begin[1] = 0; u_begin[end] = 0;
end

# ╔═╡ e2ef677f-64ef-4c56-a425-489cb38ab223
dx

# ╔═╡ 1caa046b-db63-46d0-8bbd-622bc78e75f6


# ╔═╡ bb16d432-3bde-48be-8569-511535de021c
ϵ

# ╔═╡ 3b10dd47-5ea3-4c82-b391-91c040eb0161
t_idxs = 2^14*dt:dt:tmax 

# ╔═╡ 288b7fed-766e-46b3-98ac-af1aa1534f7b
t0+ϵ+dt

# ╔═╡ c48deab7-c2e6-4e61-9b59-43c48ea4b30e
begin
	df_loss = DataFrame([:eps=>Float64[],:new_dx=>Float64[],:new_dt=>Float64[],:l1_loss=>Float64[],:l1_std=>Float64[],:l2_loss=>Float64[],:l2_std=>Float64[],:quotient=>Float64[],:x_size=>Float64[],:sigma=>String[]])
	
	drift,diff = SPDE.get_drift_diffusion()
	
	jac_prot = Tridiagonal([1.0 for i in 1:nx-1],[-2.0 for i in 1:nx],[1.0 for i in 1:nx-1]) |> sparse
	
	SDE_sparse = SPDE.SDEFunction(drift,diff;jac_prototype=jac_prot)
	
	#t0=tmax;
	#ϵ = 32dx
	time_range = ((0.0,t0+ϵ+dt))
	p=(dx,σ)
	
	E=EnsembleProblem(SDEProblem(SDE_sparse,u_begin,time_range,p))
	solution=solve(E,dtmax=dt,trajectories=1,saveat=t_idxs,progress=true,algo,maxiters=1e7)[1]
end

# ╔═╡ f02ce38d-d0a8-45ef-b448-32ba1ef694ad


# ╔═╡ 53c3bc49-1481-4bf3-971c-25666a6101b3
#SPDE.generate_solution(σ)	

# ╔═╡ 822dba46-3064-44c8-8220-f25daa8dea20
t0 + ϵ + dt

# ╔═╡ 768ad15c-aaff-4f97-9791-1479db0b8c08
Plots.plot(solution[:,end])

# ╔═╡ 2edd1c0d-0b92-47a5-951c-361bc6fa0266
4*dx

# ╔═╡ 8ee22137-fbab-4efa-b81c-932412f9f8be
begin

	x_quot = 4
	t_quot = 1
	ϵ_int = 4*dx
	#=
	df = Volatility.partial_integration(solution,dt,dx,L,tmax,x_quot,t_quot,ϵ)
	@show x_size = size(df,1)
	if x_size < 50
		return nothing
	end
	df_train,df_test = partition(df,0.8,rng=123)
	new_dx = x_quot*dx
	new_dt = t_quot*dt
	#df=DataFrame(Dict(:x=>first.(data),:y=>last.(data)))
	x_data = select(df_train,1)
	y_data = df_train.y #.* nx/t_quot
	=#
end

# ╔═╡ 443bea5a-4c24-4b67-9cdc-9f7b84495818
begin
	mat = Matrix(solution)
	@time df=SPDE.partial_integration(mat,dt,dx,L,tmax,x_quot,t_quot,ϵ_int)
end

# ╔═╡ f09756e7-4e73-466b-a2f6-4a197efe8817
Plots.scatter(df.x,df.y)

# ╔═╡ 155f5527-98a5-47c3-bc9a-c984a7ffd02d
Plots.histogram(df.x)

# ╔═╡ 26f9036a-c7f7-4be8-b073-f9c013d0f760
KNN = MLJ.@load KNNRegressor

# ╔═╡ 85f940d3-f2aa-488b-bba8-3a7678b8b7d0
begin
	truth(x) = σ(x)^2
	df_filtered = filter(row -> row.x < 15, df)
	df_train,df_test = partition(df_filtered  ,0.8,rng=123)
	x_data = select(df_filtered,1)
	y_data = df_filtered.y #.* nx/t_quot
end

# ╔═╡ 754744a9-aacd-4b2b-a255-807bcef469fe
begin
	#@show size(df)
	max_nbors = min(1500,length(df.x))
	knn=KNN()
	knn_r1=range(knn,:K,lower=20,upper=max_nbors)
	knn_r2=range(knn,:leafsize,lower=50,upper=100)
	knn_r3 = range(knn,:weights,values=[NearestNeighborModels.Uniform()])
	knn_tuned = TunedModel(model=knn,
	resampling=CV(nfolds=5),
	tuning=Grid(goal=6),
	range=[knn_r1,knn_r2,knn_r3],measure=rms,
	acceleration=CPUThreads(),
	acceleration_resampling=CPUThreads());
end

# ╔═╡ 9496c599-1fe0-47e8-a006-c99874e69af4
begin
	    mach = machine(knn_tuned,x_data,y_data)
	    #fit!(mach)
		#s=evaluate!(mach, measure=[rms,l1,l2], verbosity=0,
		#acceleration=CPUThreads())
		fit!(mach)
end

# ╔═╡ 6340b643-95e7-448b-8927-00119f3071a9
report(mach)

# ╔═╡ 16fc4b9a-a562-480c-b5d2-29abeb9b1001
MLJ.save("what",mach)

# ╔═╡ 6e8c2a59-1b4f-464c-94c0-c87f4bc8f48d
knn_best = fitted_params(mach).best_model 

# ╔═╡ 9417e33e-9596-4c4d-9bf0-fdb3886dc1c5
res=report(mach).plotting

# ╔═╡ 133835b1-3904-4f25-997e-47f3ef395b12
Plots.scatter(res.parameter_values[:,1],res.measurements)

# ╔═╡ b3458e94-487e-4160-aff1-95717b03683c
mach_predict = machine("what")

# ╔═╡ 1001c4a5-ce7d-491c-a9cf-29571aa5df9b
#Plots.scatter(x_data.x,MLJ.predict(mach_predict,x_data))

# ╔═╡ cb335dee-3802-4c41-b892-587a1d69df48
begin
	samples_idxs = sample(1:size(df_train, 1), 10000, replace=false)
	x=x_data[samples_idxs,:] 
end

# ╔═╡ 7d9dbbfc-412b-49ef-ab3a-72a4f57d5c04
begin
	Plots.scatter(x.x,MLJ.predict(mach,x))
	Plots.scatter!(x.x,σ.(x.x).^2,xlims=(0,6),ylims=(0,2))
	#Plots.scatter!(x.x,MLJ.predict(ensemble,x))
end

# ╔═╡ d426aeb9-a406-4fcd-8907-1f668f62f845
report(mach)

# ╔═╡ ddc44765-e193-4052-a0f3-f7a59360c14e
EvoSplineRegressor = @load EvoSplineRegressor pkg=EvoLinear;

# ╔═╡ d9c629a4-a067-4cd7-8b29-6e3a298e02cb
#Evo = EvoSplineRegressor(loss=:mse,L1=1e-3, L2=1e-2, nrounds=200,knots=Dict(1 => 10))

# ╔═╡ 6b81bdef-4342-4eb6-963b-cd76b92d0a4c
begin
	#mach2 = machine(Evo,x_data,y_data)
	#fit!(mach2)
end

# ╔═╡ b3586952-a00b-4e6d-a036-6915a68d0287
begin
	#Plots.plot(mach_to_func(mach2))
	#Plots.plot!(x->σ(x)^2,xlims=(0,6),ylims=(0,10))
end

# ╔═╡ ef91200e-3ec5-4b44-973f-b8e7bfc568a1


# ╔═╡ 3785c2d3-33fd-4f44-ba60-8fd836154ed8
begin
	#mach3 = machine(forest_tuned,x_data,y_data)
	#fit!(mach3)
end

# ╔═╡ 31410253-9dcc-4823-9614-2226623a629d
#fitted_params(mach3)

# ╔═╡ 6ebccf30-eb5c-4b24-96e3-6bbf22ce4745
Plots.histogram(x_data[:,1])

# ╔═╡ 163cdef1-a101-40b4-8f71-9ebe9f8cdf2a


# ╔═╡ a6a2ea19-5288-4b94-aa8a-1e8af81073e5
function mach_to_func(mach)
	est_wrapper(x) = MLJ.predict(mach,DataFrame(:x=>[x]))[1]
	return est_wrapper
end

# ╔═╡ 0243a3ed-403d-4c3d-ad94-96ae3b3646ec
adj_mean = mach_to_func(mach).(2:0.01:6) ./ truth.(2:0.01:6) |> mean

# ╔═╡ 2fd058ca-921c-4d18-b5a7-860768a029bc
begin
	Plots.scatter(x_data.x,y_data,alpha=0.05,ylims=(0,0.5),label="data")
	Plots.plot!(truth,xlims=(0,6),label="truth",linewidth=2)
	#Plots.plot!(mach_to_func(mach),markersize=0.6,linewidth=2,ylims=(0,10))
	Plots.plot!(mach_to_func(mach),linewidth=2,ylims=(0,2),label="estimate")
	#Plots.plot!(adj_fun,label="adjusted")
	#Plots.plot!(mach_to_func(mach2),markersize=0.6)
	#Plots.plot!(mach_to_func(mach3),markersize=0.6)
	
end

# ╔═╡ 8c9f4b18-d256-42a5-a084-5c7145ede197
begin
		Plots.plot(truth,xlims=(0,6),label="truth")
		Plots.plot!(mach_to_func(mach),markersize=0.6,ylims=(0,2))
end

# ╔═╡ 99d02aee-5694-46ef-a60b-a2f955ad6955
#MLJ.predict(mach3,DataFrame(:x=>[2.]))[1]

# ╔═╡ c30de2af-adae-4c5a-956c-f0f9b76835e3
mach_to_func(mach)

# ╔═╡ 5e5b209d-a7a1-4860-8430-c551abfb9ed8
SPDE.get_all_losses(mach,truth,(0,6),10000)

# ╔═╡ 3bd3f489-6135-446a-a5a3-63924ecfefd7
losses = Dict("l1"=>x->abs(x),"l2"=>x->abs(x)^2)

# ╔═╡ 4ca96151-4fc3-4cca-8054-5c449e97861d
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

# ╔═╡ eb6a4ec1-e542-4cf4-9b66-c06064bd28bb
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
		

# ╔═╡ 46e093af-f18f-43f9-8543-66c697f2e61d
best_mach, er  = get_best_machine([mach],truth,(0,6),10000,x->abs(x)^2)

# ╔═╡ 5e4b912f-064d-4736-a1c8-014eb36b6661
monte_carlo_error(mach,truth,(0,6),10000,x->abs(x))

# ╔═╡ b0cbb564-1607-4537-8b4f-828519ecbde0
function get_all_losses(mach,truth,domain,N)
	losses = Dict("l1"=>x->abs(x),"l2"=>x->abs(x)^2)
	l1 = monte_carlo_error(mach,truth,domain,N,losses["l1"])
	l2 = monte_carlo_error(mach,truth,domain,N,losses["l2"])
	return (l1,l2)
end

# ╔═╡ 9f0e4eb2-6260-4c3b-a8bb-18f923cb9165


# ╔═╡ 6bdc937b-5c59-4b42-aa3a-7a8833fdaae6
min(1,2)

# ╔═╡ 66c0ab92-4aa9-469e-b892-0a8167fdc6cb
Dict("w"=>collect(1:10))["w"]

# ╔═╡ 03fe5bcb-a11d-4508-8287-405d44853877


# ╔═╡ 1f7edc86-ab27-4af3-938d-93c4513a000f
begin
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
	
	function integrate_matrix(matrix, dx, dy)
	    # Integrate a matrix using the trapezoidal rule
	    integral = sum(matrix) * dx * dy
	    return integral
	end
end

# ╔═╡ baa72362-db61-4a1a-828c-ba6cb53fe0c9
begin
	i=0
	j=1
	@show x_quott = 1#2^i
	@show t_quott = 1#2^j
	nxx,ntt = size(solution)
	eps_factor = 8
	@show epsil = eps_factor*dx
	new_nx = nxx ÷ x_quott
	new_nt = ntt ÷ t_quott
	new_dx = x_quott*dx
	new_dt = t_quott*dt
	x_eps = (epsil ÷ new_dx) |> Int
	t_eps = (epsil ÷ new_dt) |> Int
	##t_idx = [t for t in 1:new_nt]*t_quott
	new_sol = SPDE.downsample_matrix(mat,x_quott,t_quott)#mat[x_idx,t_idx]
	Lu_clean = SPDE.L_op(new_sol,new_dt,new_dx) #.* sqrt(dx*dt) 
    Lu = Lu_clean
    #Lu = Lu * 1/sqrt(eps) * new_dx # no time
    Lu .= Lu * 1/epsil * new_dx * new_dt # with time
    buffer = 1#2^13 ÷ t_quot
    x_len,t_len = size(Lu_clean)
	#new_dx = x_quot*dx
	#new_dt = t_quot*dt
	(x_len)-x_eps-1 -(x_eps+1)
	max_x_points = (x_len)-x_eps-1 -(x_eps+1)
    num_x_samples = min(20,max_x_points)
    total_samples = min(50000,t_len*num_x_samples)
    factor = t_len*num_x_samples/total_samples |> x-> ceil(Int,x)
end

# ╔═╡ 79a321aa-73e8-4b4d-8a42-90a38cfeb2a3
abs.(Lu) .|> log

# ╔═╡ 5780c975-f31f-4fef-af56-aecc4cd5d806
downsample_matrix(mat,x_quott,t_quott)

# ╔═╡ f3c43976-8d53-40a8-a3b0-96233e92d758
size(Lu)

# ╔═╡ c09d2caa-da3b-40e7-9a2f-1a4eedac3cae
xv = 0:dx:1 

# ╔═╡ d6596528-0cea-40d8-9bf0-150a60c71251


# ╔═╡ 612ef7e6-c967-4604-af75-59688e6ca006
tv =dt:dt:ntt*dt 

# ╔═╡ c90db7e2-6d55-4743-9154-0839a1a3d74a


# ╔═╡ f78d7857-91ca-458e-824b-f56a1c045e45


# ╔═╡ b224fe00-facd-4f51-9eb5-5623e7961fb4
#Lu_int(x,t,h) = 

# ╔═╡ 93f3f0d2-03fa-47b6-b65e-3677c9886ca3
Lu

# ╔═╡ 4733f9f6-9a40-4b54-bb0e-8c9deeac6a53
begin
	m_test = rand(100,100)
	views_test = view(m_test,40:40+10,40:40+10)
	l1,l2 = size(views_test)
	trapz((1:l1,1:l2),views_test)
end

# ╔═╡ 5dd4480f-f832-4a0a-ab61-ae63a3219e60
sum(views_test)

# ╔═╡ 75035228-51f1-4026-aed9-15941a7c31e6
tr

# ╔═╡ ed9232c4-6c74-45b4-910f-7b9423db99fb
(Lu[64:64+x_eps,300:300+t_eps]) |> sum |> x->x^2

# ╔═╡ a4265f82-23a0-406a-bafd-9e61b6b22dbf


# ╔═╡ 57145b2b-89a0-47e8-9705-da257375c42d


# ╔═╡ 082fc769-4f53-411d-bb30-5c70384334e7
size(new_sol)

# ╔═╡ 20091daa-67f6-4a1d-9452-9446bd1a8949
new_dx

# ╔═╡ cd48cf11-a71a-4e18-aa1b-abdb0605283c
Lu[1:1+x_eps,1:1+t_eps] |> sum

# ╔═╡ f931563c-5702-4242-be49-f4984ba5f6c7
solution

# ╔═╡ cbfe8798-7ccd-4a8e-bab8-a39188a2ccf8
Lu[5,100:1:end-100]

# ╔═╡ e28a1d15-bc52-44d2-a25d-151270036074


# ╔═╡ ce7d0f1f-b87d-410d-b2ed-bc922daa7b19
(epsil / new_dt) #|> floor #|> Int

# ╔═╡ 9318d024-f9f9-425d-9712-4567503a7fe4
(epsil / new_dx) #|> floor #|> Int

# ╔═╡ 75fc99bd-bd17-4409-afa7-61502cf49a04
abs.(Lu[1,:]) .|> log

# ╔═╡ 655d1b65-3c28-4e65-a88f-fac09f205210


# ╔═╡ 6a33a3c9-2237-4fa0-9d2d-b934e3795a87
4*dx

# ╔═╡ 15670eb1-06d6-4411-8f45-fd60d6e4ee4c
t_eps

# ╔═╡ 05695004-aef2-4a25-bd5e-8e19f65730ad
Plots.scatter(Lu[3,:])

# ╔═╡ fe5d1f72-cb55-4cba-adc8-7cfb2b30134c


# ╔═╡ 2b2a3f5e-5565-42bf-a9d1-9cd58d537768
x_quot

# ╔═╡ 3b87c33f-06f5-47e2-8bca-7e5537cd0841
1/2^(1/12)

# ╔═╡ e142d48d-3939-4247-bf62-ff431da9bdf0
x_eps+1:x_len-x_eps-1

# ╔═╡ 6a4cdc24-0c3f-479d-8034-a4db376d69c8
rand_x = sample(x_eps+1:x_len-x_eps-1,num_x_samples,replace=false)

# ╔═╡ a596f292-7bbd-4e4e-99d5-40983ccdc306
solution[:,end] |> Plots.plot

# ╔═╡ 9c7eae35-ab7e-4918-9498-dfee5a241924
dff=SPDE.partial_integration(Matrix(solution),dt,dx,L,tmax,x_quot,t_quot,32*dx)

# ╔═╡ 8b5e2999-56b0-4f00-88e5-8510f6d1513b
dff.y |> median

# ╔═╡ 909af3f6-3742-41f1-b4f1-b5715b514cb2
DataFrame(rand(2),rand(2))

# ╔═╡ 0efecc3c-de11-4fa2-ade0-010e0eb4593c
(x_len)-x_eps-1 -(x_eps+1)

# ╔═╡ 93d99f57-42ec-45a3-8065-0745c0e83f8b
epsil/new_dx

# ╔═╡ 3588295f-1dd8-44a4-b372-4ec62e25397d
x_len

# ╔═╡ 41c62873-2962-42d9-acb5-9627696243e7


# ╔═╡ 708d5399-a742-4d0d-af16-c56737472795
x_eps

# ╔═╡ 8033715c-2cdb-4c54-9964-005700afe20e
t_eps

# ╔═╡ 5f7e878a-1c12-4d45-8709-1363f01b737a
size(Lu)

# ╔═╡ 8242f7b1-2500-4155-95d6-e71dba7fa9b1
size(Lu)

# ╔═╡ da672d3e-ba57-408e-b735-da470e3fd291
nt ÷ 8

# ╔═╡ 3c7a1844-c258-4a3a-923c-75728df56d80
nt

# ╔═╡ 2b13e402-3166-4fd7-924c-da0d72885fbb
ntt

# ╔═╡ 77b92db2-9cf4-45af-b7a0-f639cd0e3c69
32768*8

# ╔═╡ 70507c23-9e3a-404e-9772-d0413baf7b9d
mat

# ╔═╡ a53a6b1f-d7b8-472f-997f-9ce4e802a690
ddd = dicts[1]

# ╔═╡ 1703f536-3eca-489a-9f4d-192fa652753b
datadir("simulations", savename(ddd, "jld2"))

# ╔═╡ e6f4bb9a-9529-4119-aa75-7620d4f5f432


# ╔═╡ df238cb5-104f-435a-b618-91e54a6c71a4
MLJ.save(datadir("machines",savename(dicts[1],"jld2")),machs)

# ╔═╡ f895f565-65a3-4e02-ae0f-22459d5a96a9
par=Dict("sigma"=>"sigma2","xquot"=>1,"tquot"=>8,"epsilon"=>2^6*dx,"sample"=>1)

# ╔═╡ f320b279-684d-45f1-ad23-59b8a9cd7246
#machs,fs,tr,df_main = SPDE.main_exp(par)

# ╔═╡ 8548b927-5de1-49c2-9875-7427ecfbe081
df

# ╔═╡ 133abe19-ffd0-4af5-857b-6dae771750da
begin
	Plots.plot(mach_to_func(machs),xlims=(0,6))
	Plots.plot!(tr)
end

# ╔═╡ 04c7845b-c4a0-43a4-924f-9ffde85defda
gg(x) = mach_to_func(machs)(x)/tr(x)

# ╔═╡ dd5bac94-df1c-4fa2-97b3-cd743ed088de
Plots.histogram(df_main.x)

# ╔═╡ 941c5017-da4f-4e13-8643-c64b0ed568a4
gg.(1:0.01:4) |> median

# ╔═╡ 47f31735-35bb-4efb-a2a5-de10274f28a3
(2*dx)^(0.8)

# ╔═╡ 2f5c65d6-6bfa-4cfc-8893-7596b78537c5
Plots.plot(gg,xlims=(1,6),ylims=(0,1))

# ╔═╡ 62957f8a-2a07-4a1d-a32c-72e93386e787
report(machs)

# ╔═╡ cc52656f-79c3-4e5c-b7f2-f5583af9d8c2
1/(2dx)

# ╔═╡ 84423e81-e64b-4947-a002-ae0ea33aae26
(2*dx)

# ╔═╡ f33ee69f-4f95-4f4c-85ab-e4fa207a7389
2^5

# ╔═╡ 8f30519f-3029-4a00-b742-7ee09f56809c
begin
	SPDE_parameters = Dict(
		"epsilon" => [2^i*dx for i in 2:4],
		"xquot" => [2^i for i in 2:2],
		"tquot" => [2^i for i in 0:7],
		"sigma" => ["sigma2"]
		#"sample" => [collect(1:10)]
	)
	#dicts = dict_list(SPDE_parameters)
	savename(dicts[1])
end

# ╔═╡ 12240a6a-424d-4f1b-9ea2-17a70f285e08


# ╔═╡ d6c9451c-0c24-467e-8124-ac57bc4bbb17
begin
	SPDE_parameters_1 = Dict(
	"epsilon" => [2^i*dx for i in 1:6], #cq + eps \leq 6
	"xquot" => [2^i for i in 0:0], #5
	"tquot" => [2^i for i in 0:7], #7
	"sigma" => ["sigma1"],
	"sample" => collect(1:5)
	)
	
	SPDE_parameters_2 = Dict(
		"epsilon" => [2^i*dx for i in 1:5], #cq + eps \leq 6
		"xquot" => [2^i for i in 1:1], #5
		"tquot" => [2^i for i in 0:7], #7
		"sigma" => ["sigma1"],
		"sample" => collect(1:5))
	SPDE_parameters_3 = Dict(
		"epsilon" => [2^i*dx for i in 2:5], #cq + eps \leq 6
		"xquot" => [2^i for i in 2:2], #5
		"tquot" => [2^i for i in 0:7], #7
		"sigma" => ["sigma1"],
		"sample" => collect(1:5))
	SPDE_parameters_4 = Dict(
		"epsilon" => [2^i*dx for i in 3:4], #cq + eps \leq 6
		"xquot" => [2^i for i in 3:3], #5
		"tquot" => [2^i for i in 0:7], #7
		"sigma" => ["sigma1"],
		"sample" => collect(1:5))
	SPDE_parameters_5 = Dict(
		"epsilon" => [2^i*dx for i in 4:4], #cq + eps \leq 6
		"xquot" => [2^i for i in 4:4], #5
		"tquot" => [2^i for i in 0:7], #7
		"sigma" => ["sigma1"],
		"sample" => collect(1:5))
	long_params = vcat(dict_list.([SPDE_parameters_1,SPDE_parameters_2,SPDE_parameters_3,SPDE_parameters_4,SPDE_parameters_5])...)
end

# ╔═╡ 4d19f17b-06e0-438e-ae97-d34529390e12
dict_list(SPDE_parameters_1) |> typeof

# ╔═╡ 86d1d92a-0d25-4759-b191-2779bd935232
0.3/dx

# ╔═╡ a1366b22-eedd-4b3c-b263-52f14a201c15
2^4

# ╔═╡ 0c81e345-e3f9-46f2-8bb8-6a7a394b253a
2^2*2^5

# ╔═╡ 6a3418a1-a2fc-4b1e-96aa-4e7e324bb785


# ╔═╡ dbeced1f-e38b-4fbd-9879-bcde79b41e70
(2*8+8+4*8+3*8)*5

# ╔═╡ 50b9a52d-af0a-47cb-bf54-378f33caba9c
2^8

# ╔═╡ 6f853540-cfca-4730-9f40-67ce75b00e7c
2^5

# ╔═╡ defa8cad-88b9-442b-b774-356bdb787ae1
mat

# ╔═╡ b4b18740-d7e9-428d-b04c-2de36ef0f1a3
begin
	testa = Dict(
		"epsilon_factor" => [i for i in 1:5],
		"xquot" => [2^i for i in 1:1],
		"tquot" => [2^i for i in 0:0],
		"sigma" => ["sigma2"],
		"sample" => [collect(1:10)]
	)
	test_dict = dict_list(testa)
	savename(test_dict[1])
end

# ╔═╡ 3a51b7f9-b549-4f79-b991-44c1e141ba53
dx/dt

# ╔═╡ 499d79fc-c8e4-430f-a94c-9f475136626b


# ╔═╡ 85e384b2-7096-4c84-9cfb-9ba01a26a43a
#=for (i, d) in enumerate(long_params)
    mach,f,tr,df_main = SPDE.main_exp(d)
	println(d)
    wsave(datadir("simulations", savename(d, "jld2")), f)
	MLJ.save(datadir("machines",savename(d,"jld2")),mach)
end
=#


# ╔═╡ 08eacc34-81c0-4774-8c0a-655f658e09a0


# ╔═╡ 2955aff4-2bff-455e-9695-a81b452a6c13


# ╔═╡ 078d216e-191d-4489-a141-2219e82b526b



# ╔═╡ 9186853a-9267-4bc6-9a83-036fc4fad673
#=for (i, d) in enumerate(long_params)
    mach,f,tr,df_main = SPDE.main_exp(d)
	println(d)
    #wsave(datadir("simulations", savename(d, "jld2")), f)
	MLJ.save(datadir("mail_plots",savename(d,"jld2")),mach)
end
=#

# ╔═╡ 420b2ccd-76d5-4cd5-85e5-b6294586dada
df_final = collect_results(datadir("simulations"))

# ╔═╡ 1478242e-7668-4172-a8cc-f6dbcdaed10b


# ╔═╡ 78191406-7b7e-4f27-b3f8-67e55dd03b01
# ╠═╡ disabled = true
#=╠═╡
begin
	plt=data(df_final) * mapping(:xquot,:l1,layout=:epsilon=>nonnumeric) * visual(Scatter)
	draw(plt)
end
  ╠═╡ =#

# ╔═╡ f8fa8b0a-f273-4009-8ec7-121ee213937f
begin
	
	
	# Define the path where your .jld2 files are located
	path_to_files =datadir("machines")
	
	# Get a list of .jld2 files in the directory
	files = filter(f -> occursin(r"\.jld2$", f), readdir(path_to_files, join=true))
	
	# Initialize a dictionary to store the modified machines
	# The keys will be the original file names (or a modified version of them), and the values will be the modified machines
	modified_machines = Dict()
	
	# Loop over the files, process each machine, and store the result in the dictionary
	for file_path in files
		println(file_path)
		# Load the machine
		mach = machine(file_path)
		
		# Apply your function to the machine
		modified_machine = mach_to_func(mach)
		
		# Extract a useful key from the file path (e.g., the base name without extension)
		key = split(basename(file_path), ".jl")[1]
		println(key)
		
		# Store the modified machine in the dictionary
		modified_machines[key] = modified_machine
	end

end

# ╔═╡ 24efc612-3b74-404c-bbd3-31b22d655f56
keys_list = collect(keys(modified_machines))

# ╔═╡ e0f1085e-6791-4e85-9747-6944c840cc85
begin
	plt2=Plots.plot()
	for (key, func) in modified_machines
		Plots.plot!(func,xlims=(0,6),ylims=(0,10),label="")  # Plot the resulting y-values against x-values
	end
end

# ╔═╡ 41f31996-b5ac-4500-a331-df691e54be59
function get_function(tquot, xquot, epsilon, sigma, machines_dict)
    # Construct the key based on the parameters
    # Adjust the format as needed to match your keys
    key = "epsilon=$(epsilon)_sigma=$(sigma)_tquot=$(tquot)_xquot=$(xquot)"
    println(key)
    # Retrieve and return the function from the dictionary
    if haskey(machines_dict, key)
        return machines_dict[key]
    else
        error("Function with specified parameters not found.")
    end
end

# ╔═╡ af068abe-6a23-4d1d-adf1-daca8fa567ff
function filter_filenames(folder_path, epsilon, sigma, tquot, xquot)
    # Construct the pattern to match the files based on specified variables
    pattern = "epsilon=$(epsilon)_sample=*_sigma=$(sigma)_tquot=$(tquot)_xquot=$(xquot).jld2"
    
    # Use Glob to find files matching the pattern
    matching_files = glob(pattern, folder_path)
    
    return matching_files
end

# ╔═╡ c38c520d-cfd3-4b1d-865a-38b71c244204
epsilons=[0.125,0.0626,0.0313,0.0157,0.00783,0.00196]

# ╔═╡ b557f31f-d72e-4e3f-97e7-cbb21413d57c
path_mail = datadir("mail_machines")

# ╔═╡ e00cd98e-8490-4af4-aca1-402c3ae53f79


# ╔═╡ e299e1ea-a00a-4dc4-8e36-cb91a254b5e9
begin
	df_quot = DataFrame(:eps=>Float64[],:tq=>Float64[],:xq=>Float64[],:quot=>Float64[],
	:std_est=>Float64[])
	for epsilon in epsilons
		#sample_points=mean_quot.(collect(2:0.01:6)) #|> mean
		Threads.@threads for tq in [2^i for i in 0:7]
			for xq in [2^i for i in 0:5]
				path_samples = filter_filenames(path_to_files,string(epsilon),"sigma1",string(tq),string(xq))
				if isempty(path_samples)
					break
				else
					fun_samples = machine.(path_samples) .|> mach_to_func
					mean_func(x) = [fun(x) for fun in fun_samples] |> mean
					mean_quot(x) = mean_func(x)/truth(x)
					quot_array=mean_quot.(collect(2:0.01:6)) #|> mean
					quot_est = mean(quot_array)
					std_est = std(quot_array)
					
					push!(df_quot,(epsilon,tq,xq,quot_est,std_est))
				end
						
			end
		end
	end			
end

# ╔═╡ bf46e435-f483-448c-a6a1-6011fc5a78af
df_quot

# ╔═╡ b393139a-b734-441e-a6e8-f0f8fe68b33c
begin
	axis = (type = Axis3, width = 600, height = 600)
	data(df_quot)*mapping(:xq=>log,:tq=>log,:eps=>log,color=:quot=>log) |> x->draw(x;axis=axis)
end

# ╔═╡ 71db0b3d-6af8-4019-a31f-12c532a3bc43
df_quot.var = df_quot.std_est ./ df_quot.quot

# ╔═╡ 3660b185-45b9-4bd4-9007-4d7a8014bd51
Plots.histogram(df_quot.var)

# ╔═╡ 3bc95456-b4b3-4d62-b022-5e72b8be9154
data(df_quot)*mapping(:tq=>log,:quot=>log,layout=:xq=>nonnumeric,color=:eps=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ e41a9ff9-0001-49d2-9f89-b60e42e74e81
df_quot.quot |> Plots.scatter

# ╔═╡ fed4e2b5-a962-4a69-975a-8ae80296c523
data(df_quot)*mapping(:tq=>log,:quot=>log,layout=:xq=>nonnumeric,color=:eps=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ b2332c65-5054-4b5f-91b0-63abd18fdebd
(8dx)^(0.67)

# ╔═╡ 68d35ad6-a77e-4eda-af4f-c914dc69fd25
data(df_quot)*mapping(:eps=>log,:quot=>log,layout=:xq=>nonnumeric,color=:tq=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ e4571a77-7358-4b0c-959b-0540ae5a0dec
df_quot

# ╔═╡ e60a7f31-bbf4-4fc7-9ec8-c247036bd303
df_quot.prod = df_quot.tq .* df_quot.xq

# ╔═╡ 66aa977f-a8e9-447e-b755-67569e23b25f
df_quot.xt_quot = df_quot.tq ./ df_quot.xq

# ╔═╡ bdbc1b33-d018-4bca-b7dc-d6b6188c047b
df_quot

# ╔═╡ e427fe79-5d08-45fc-9fc2-9706449ee1e7
data(df_quot)*mapping(:scaling=>log,:quot=>log,color=:tq=>log,layout=:xq=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ 26e94c39-ff53-41e1-898b-22e1ffa29cab
data(df_quot)*mapping(:prod=>log,:quot=>log,layout=:xq=>nonnumeric,color=:eps)*visual(Scatter) |> draw

# ╔═╡ 9e585547-192e-427f-ba19-715077d8cf1b
df_quot

# ╔═╡ 75fc4288-f433-40a7-bcd0-5bab7d14ed3b
data_=log.(df_quot.quot) ./ log.(df_quot.prod)

# ╔═╡ f7a6998f-bdb4-4c86-9883-97742826b18b
filtered_data = filter(x -> x != -Inf, data_) .|> exp |> median

# ╔═╡ c68a4744-2fa9-4b67-8ae7-0a10e1c2e078
begin
	max_value, max_index = findmax(df_quot.quot)
	
	# Retrieve the row with the maximal :x value
	row_with_max_x = df_quot[max_index, :]
end

# ╔═╡ 5102fa56-b8f2-4817-b10f-fc4a4fd3b645
@bind id_sort PlutoUI.Slider(1:size(df_quot,1))

# ╔═╡ 9ea3bbdf-70c6-414e-8bc7-a3f4709f0198
begin
	sorted_df = sort(df_quot, :quot)
	
	# Retrieve the second last row, which corresponds to the second largest value
	row_with_second_largest_quot = sorted_df[id_sort, :]
end

# ╔═╡ bd13a315-a8f9-49b3-84cb-0b9245b0bca1
32*dx

# ╔═╡ 055905f6-0e27-4635-837e-d2dcfe9122d8


# ╔═╡ 5917870f-453f-42b8-94a2-f2e39ff01f0f
@bind xq Select([2^i for i in 0:5])

# ╔═╡ 904c5b0a-1ee1-46f0-aee1-867169ffb096
@bind tqs Select([2^i for i in 0:7])

# ╔═╡ acf51970-f888-4741-a4c0-c6c562a9db15
@bind eps Select(epsilons)

# ╔═╡ 75079761-d1c2-48df-aa6d-531bf66186f5
begin
	path_samples = filter_filenames(path_to_files,string(eps),"sigma2",string(tqs),string(xq))
	fun_samples = machine.(path_samples) .|> mach_to_func
end

# ╔═╡ 2039f019-7ffc-4c6a-ae33-05e2c24fc54b
path_samples

# ╔═╡ 99673b9d-819e-4e32-ac36-80bdae1aa888
path_samples

# ╔═╡ 9dffdc1c-38d3-4b21-adb8-4ec7e7ea3d62
mean_func(x) = [fun(x) for fun in fun_samples] |> mean

# ╔═╡ ab253287-e6e7-4289-8a3c-32b09ee485ad
df_quot.scaling = df_quot.tq .* df_quot.xq ./ eps

# ╔═╡ 52a05fd9-ce3b-4d8f-8ef4-05ba62e83a8b
32*dx

# ╔═╡ 0ad816c2-55d9-4461-8ef9-fa7e959ea8e8
std

# ╔═╡ 1ae356ee-f287-41e0-bb92-47b54dbd36c5
a = rand(10)

# ╔═╡ 5328ff94-d492-4fde-9401-2e1b54405345
begin
	idxs = 1:10 |> collect
	scaled = 3*idxs
end

# ╔═╡ 9b32c621-f694-47cf-a945-0efc3b3e47a4
truth

# ╔═╡ dece773e-1fe4-4c1d-9e4e-c1b32dad107b
Plots.scatter(df_quot.quot,ylims=(0,2))

# ╔═╡ a300a6ed-9955-4f83-b262-62ce10a540ed


# ╔═╡ b8314e1c-fda4-442b-8e79-b867dd8d2533
df_quot.quot |> median

# ╔═╡ 4b4f0d7f-e977-4415-8f3f-4a1f78478845
#=begin
	plot_truth(x) = (0.5x)^2
	mean_quot(x) = mean_func(x)/plot_truth(x)
	quot_array=mean_quot.(collect(2:0.01:6)) #|> mean
	quot_est = mean(quot_array)
	std_est = std(quot_array)
	adjusted_estimate(x) = mean_func(x)/quot_est

	plots_samples = Plots.plot()
	for fun in fun_samples
		Plots.plot!(fun,xlims=(2,6),label="",alpha=0.5)
	end
	Plots.plot!(plot_truth,label=truth,color=:black,linewidth=2.,)
	Plots.plot!(adjusted_estimate,label="adjusted",linewidth=2.,color=:blue)
	Plots.plot!(mean_func,label="mean estimate",linewidth=2,color=:red)
	Plots.plot!(mean_quot,ylims=(0,10),label="quotient")

	plots_samples
end
=#

# ╔═╡ 6d5d9692-25c3-42f2-a45f-336c90585789
println(quot_est)

# ╔═╡ 30921264-6a59-4d35-8b7d-3e47445588be
2*dx

# ╔═╡ 8408a1b8-5899-4a1e-a1fa-9882600c446d
#=begin
	begin
			mail_params = Dict(
				"epsilon" => [2*dx,4*dx,8*dx,16*dx,32*dx], #cq + eps \leq 6
				"xquot" => [2], #5
				"tquot" => [4], #7
				"sigma" => ["sigma3"],
				"sample" => collect(1:5))
			mail_dict = dict_list(mail_params)
	end
	for (i, d) in enumerate(mail_dict)
	    mach,f,tr,df_main = SPDE.main_exp(d)
		println(d)
	    #wsave(datadir("simulations", savename(d, "jld2")), f)
		MLJ.save(datadir("mail_machines",savename(d,"jld2")),mach)
	end
end
=#

# ╔═╡ 4fc4e5f5-acb3-4fb4-bcc8-4486c674b266


# ╔═╡ b8890dbb-5ed7-44d3-9f0a-bbc9e304a1e0


# ╔═╡ 6961d1b0-0f7e-49a1-aa7c-1e2cc93c37db
16*dx

# ╔═╡ 8f1eed65-2ff2-4b42-a2f4-10c2ac359bd3
epsilons

# ╔═╡ 9e80322b-6ced-48a5-8832-445e3902b4a7


# ╔═╡ 6d0acdd8-a20e-4522-b7f9-1c942600d336


# ╔═╡ ef16799f-45d2-4dc2-8c85-9e89941f6d0c


# ╔═╡ 9bebfb65-3b2d-4441-91b7-c26293ccf901


# ╔═╡ 382cda6b-0181-4762-83cf-7e5a85eec1f3
begin

		mail_funcs=filter_filenames(datadir("mail_machines"), string(epsilons[i]), "sigma2", string(8), string(2)) .|> machine .|> mach_to_func
		mean_mail(x) = [fun(x) for fun in mail_funcs] |> mean
	
		
		mail_truth(x) = sin(x)^2
		mean_quot(x) = mean_mail(x)/mail_truth(x)
		quot_array=mean_quot.(collect(3.5:0.01:5)) #|> mean
		quot_est = median(quot_array) *1.2
		std_est = std(quot_array)
		adjusted_estimate(x) = mean_mail(x)/quot_est
	
		plots_samples = Plots.plot()
		for fun in mail_funcs
			Plots.plot!(fun,xlims=(2,6),label="",alpha=0.5)
		end
		Plots.plot!(mail_truth,label=L"σ^2",color=:black,linewidth=2.,)
		Plots.plot!(adjusted_estimate,label="adjusted regression",linewidth=2.,color=:blue)
	end
	Plots.plot!(mean_mail,label="regression mean",linewidth=2,color=:red)
	#Plots.plot!(mean_quot,ylims=(0,10),label="regression/sigma^2")

end


# ╔═╡ 4a0820db-4ccb-4821-af26-d79782f8c7c5


# ╔═╡ 2f182a73-dcd3-4694-a028-f3102b74232b
#Plots.gif(anim_1,fps=1)

# ╔═╡ Cell order:
# ╠═f612d560-9988-4d3b-a605-8d145d7f434e
# ╠═6a92067c-958c-48a6-9e64-b94edfb3bb45
# ╠═6c541e1a-dd2f-11ee-1948-03057f232327
# ╠═a0576e78-e142-4fb6-9b96-c9edc91d1707
# ╠═e2ef677f-64ef-4c56-a425-489cb38ab223
# ╠═85463824-794f-4f9c-85b6-31ecc2f02b06
# ╠═1caa046b-db63-46d0-8bbd-622bc78e75f6
# ╠═bb16d432-3bde-48be-8569-511535de021c
# ╠═3b10dd47-5ea3-4c82-b391-91c040eb0161
# ╠═288b7fed-766e-46b3-98ac-af1aa1534f7b
# ╠═c48deab7-c2e6-4e61-9b59-43c48ea4b30e
# ╠═f02ce38d-d0a8-45ef-b448-32ba1ef694ad
# ╠═53c3bc49-1481-4bf3-971c-25666a6101b3
# ╠═822dba46-3064-44c8-8220-f25daa8dea20
# ╠═768ad15c-aaff-4f97-9791-1479db0b8c08
# ╠═2edd1c0d-0b92-47a5-951c-361bc6fa0266
# ╠═8ee22137-fbab-4efa-b81c-932412f9f8be
# ╠═443bea5a-4c24-4b67-9cdc-9f7b84495818
# ╠═f09756e7-4e73-466b-a2f6-4a197efe8817
# ╠═155f5527-98a5-47c3-bc9a-c984a7ffd02d
# ╠═26f9036a-c7f7-4be8-b073-f9c013d0f760
# ╠═85f940d3-f2aa-488b-bba8-3a7678b8b7d0
# ╠═754744a9-aacd-4b2b-a255-807bcef469fe
# ╠═6340b643-95e7-448b-8927-00119f3071a9
# ╠═9496c599-1fe0-47e8-a006-c99874e69af4
# ╠═16fc4b9a-a562-480c-b5d2-29abeb9b1001
# ╠═6e8c2a59-1b4f-464c-94c0-c87f4bc8f48d
# ╠═9417e33e-9596-4c4d-9bf0-fdb3886dc1c5
# ╠═133835b1-3904-4f25-997e-47f3ef395b12
# ╠═b3458e94-487e-4160-aff1-95717b03683c
# ╠═1001c4a5-ce7d-491c-a9cf-29571aa5df9b
# ╠═cb335dee-3802-4c41-b892-587a1d69df48
# ╠═7d9dbbfc-412b-49ef-ab3a-72a4f57d5c04
# ╠═d426aeb9-a406-4fcd-8907-1f668f62f845
# ╠═ddc44765-e193-4052-a0f3-f7a59360c14e
# ╠═d9c629a4-a067-4cd7-8b29-6e3a298e02cb
# ╠═6b81bdef-4342-4eb6-963b-cd76b92d0a4c
# ╠═b3586952-a00b-4e6d-a036-6915a68d0287
# ╠═ef91200e-3ec5-4b44-973f-b8e7bfc568a1
# ╠═3785c2d3-33fd-4f44-ba60-8fd836154ed8
# ╠═31410253-9dcc-4823-9614-2226623a629d
# ╠═6ebccf30-eb5c-4b24-96e3-6bbf22ce4745
# ╠═0243a3ed-403d-4c3d-ad94-96ae3b3646ec
# ╠═163cdef1-a101-40b4-8f71-9ebe9f8cdf2a
# ╠═2fd058ca-921c-4d18-b5a7-860768a029bc
# ╠═8c9f4b18-d256-42a5-a084-5c7145ede197
# ╠═a6a2ea19-5288-4b94-aa8a-1e8af81073e5
# ╠═99d02aee-5694-46ef-a60b-a2f955ad6955
# ╠═eb6a4ec1-e542-4cf4-9b66-c06064bd28bb
# ╠═c30de2af-adae-4c5a-956c-f0f9b76835e3
# ╠═46e093af-f18f-43f9-8543-66c697f2e61d
# ╠═5e4b912f-064d-4736-a1c8-014eb36b6661
# ╠═b0cbb564-1607-4537-8b4f-828519ecbde0
# ╠═5e5b209d-a7a1-4860-8430-c551abfb9ed8
# ╠═3bd3f489-6135-446a-a5a3-63924ecfefd7
# ╠═4ca96151-4fc3-4cca-8054-5c449e97861d
# ╠═9f0e4eb2-6260-4c3b-a8bb-18f923cb9165
# ╠═5a72d3dc-8d13-4aaa-80c9-e0ab9cf35638
# ╠═6bdc937b-5c59-4b42-aa3a-7a8833fdaae6
# ╠═66c0ab92-4aa9-469e-b892-0a8167fdc6cb
# ╠═03fe5bcb-a11d-4508-8287-405d44853877
# ╠═79a321aa-73e8-4b4d-8a42-90a38cfeb2a3
# ╠═1f7edc86-ab27-4af3-938d-93c4513a000f
# ╠═5780c975-f31f-4fef-af56-aecc4cd5d806
# ╠═f3c43976-8d53-40a8-a3b0-96233e92d758
# ╠═6d4016e8-f799-4fea-b0a5-dab23f467848
# ╠═baa72362-db61-4a1a-828c-ba6cb53fe0c9
# ╠═c09d2caa-da3b-40e7-9a2f-1a4eedac3cae
# ╠═d6596528-0cea-40d8-9bf0-150a60c71251
# ╠═612ef7e6-c967-4604-af75-59688e6ca006
# ╠═c90db7e2-6d55-4743-9154-0839a1a3d74a
# ╠═f78d7857-91ca-458e-824b-f56a1c045e45
# ╠═b224fe00-facd-4f51-9eb5-5623e7961fb4
# ╠═93f3f0d2-03fa-47b6-b65e-3677c9886ca3
# ╠═4733f9f6-9a40-4b54-bb0e-8c9deeac6a53
# ╠═5dd4480f-f832-4a0a-ab61-ae63a3219e60
# ╠═75035228-51f1-4026-aed9-15941a7c31e6
# ╠═ed9232c4-6c74-45b4-910f-7b9423db99fb
# ╠═a4265f82-23a0-406a-bafd-9e61b6b22dbf
# ╠═57145b2b-89a0-47e8-9705-da257375c42d
# ╠═082fc769-4f53-411d-bb30-5c70384334e7
# ╠═20091daa-67f6-4a1d-9452-9446bd1a8949
# ╠═cd48cf11-a71a-4e18-aa1b-abdb0605283c
# ╠═f931563c-5702-4242-be49-f4984ba5f6c7
# ╠═cbfe8798-7ccd-4a8e-bab8-a39188a2ccf8
# ╠═e28a1d15-bc52-44d2-a25d-151270036074
# ╠═ce7d0f1f-b87d-410d-b2ed-bc922daa7b19
# ╠═9318d024-f9f9-425d-9712-4567503a7fe4
# ╠═75fc99bd-bd17-4409-afa7-61502cf49a04
# ╠═655d1b65-3c28-4e65-a88f-fac09f205210
# ╠═6a33a3c9-2237-4fa0-9d2d-b934e3795a87
# ╠═15670eb1-06d6-4411-8f45-fd60d6e4ee4c
# ╠═05695004-aef2-4a25-bd5e-8e19f65730ad
# ╠═fe5d1f72-cb55-4cba-adc8-7cfb2b30134c
# ╠═2b2a3f5e-5565-42bf-a9d1-9cd58d537768
# ╠═3b87c33f-06f5-47e2-8bca-7e5537cd0841
# ╠═e142d48d-3939-4247-bf62-ff431da9bdf0
# ╠═6a4cdc24-0c3f-479d-8034-a4db376d69c8
# ╠═a596f292-7bbd-4e4e-99d5-40983ccdc306
# ╠═9c7eae35-ab7e-4918-9498-dfee5a241924
# ╠═8b5e2999-56b0-4f00-88e5-8510f6d1513b
# ╠═909af3f6-3742-41f1-b4f1-b5715b514cb2
# ╠═0efecc3c-de11-4fa2-ade0-010e0eb4593c
# ╠═93d99f57-42ec-45a3-8065-0745c0e83f8b
# ╠═3588295f-1dd8-44a4-b372-4ec62e25397d
# ╠═41c62873-2962-42d9-acb5-9627696243e7
# ╠═708d5399-a742-4d0d-af16-c56737472795
# ╠═8033715c-2cdb-4c54-9964-005700afe20e
# ╠═5f7e878a-1c12-4d45-8709-1363f01b737a
# ╠═8242f7b1-2500-4155-95d6-e71dba7fa9b1
# ╠═da672d3e-ba57-408e-b735-da470e3fd291
# ╠═3c7a1844-c258-4a3a-923c-75728df56d80
# ╠═2b13e402-3166-4fd7-924c-da0d72885fbb
# ╠═77b92db2-9cf4-45af-b7a0-f639cd0e3c69
# ╠═70507c23-9e3a-404e-9772-d0413baf7b9d
# ╠═a53a6b1f-d7b8-472f-997f-9ce4e802a690
# ╠═1703f536-3eca-489a-9f4d-192fa652753b
# ╠═e6f4bb9a-9529-4119-aa75-7620d4f5f432
# ╠═df238cb5-104f-435a-b618-91e54a6c71a4
# ╠═f895f565-65a3-4e02-ae0f-22459d5a96a9
# ╠═f320b279-684d-45f1-ad23-59b8a9cd7246
# ╠═8548b927-5de1-49c2-9875-7427ecfbe081
# ╠═133abe19-ffd0-4af5-857b-6dae771750da
# ╠═04c7845b-c4a0-43a4-924f-9ffde85defda
# ╠═dd5bac94-df1c-4fa2-97b3-cd743ed088de
# ╠═941c5017-da4f-4e13-8643-c64b0ed568a4
# ╠═47f31735-35bb-4efb-a2a5-de10274f28a3
# ╠═2f5c65d6-6bfa-4cfc-8893-7596b78537c5
# ╠═62957f8a-2a07-4a1d-a32c-72e93386e787
# ╠═cc52656f-79c3-4e5c-b7f2-f5583af9d8c2
# ╠═84423e81-e64b-4947-a002-ae0ea33aae26
# ╠═f33ee69f-4f95-4f4c-85ab-e4fa207a7389
# ╠═8f30519f-3029-4a00-b742-7ee09f56809c
# ╠═12240a6a-424d-4f1b-9ea2-17a70f285e08
# ╠═4d19f17b-06e0-438e-ae97-d34529390e12
# ╠═d6c9451c-0c24-467e-8124-ac57bc4bbb17
# ╠═86d1d92a-0d25-4759-b191-2779bd935232
# ╠═a1366b22-eedd-4b3c-b263-52f14a201c15
# ╠═0c81e345-e3f9-46f2-8bb8-6a7a394b253a
# ╠═6a3418a1-a2fc-4b1e-96aa-4e7e324bb785
# ╠═dbeced1f-e38b-4fbd-9879-bcde79b41e70
# ╠═50b9a52d-af0a-47cb-bf54-378f33caba9c
# ╠═6f853540-cfca-4730-9f40-67ce75b00e7c
# ╠═defa8cad-88b9-442b-b774-356bdb787ae1
# ╠═b4b18740-d7e9-428d-b04c-2de36ef0f1a3
# ╠═3a51b7f9-b549-4f79-b991-44c1e141ba53
# ╠═499d79fc-c8e4-430f-a94c-9f475136626b
# ╠═85e384b2-7096-4c84-9cfb-9ba01a26a43a
# ╠═08eacc34-81c0-4774-8c0a-655f658e09a0
# ╠═2955aff4-2bff-455e-9695-a81b452a6c13
# ╠═078d216e-191d-4489-a141-2219e82b526b
# ╠═9186853a-9267-4bc6-9a83-036fc4fad673
# ╠═420b2ccd-76d5-4cd5-85e5-b6294586dada
# ╠═1478242e-7668-4172-a8cc-f6dbcdaed10b
# ╠═78191406-7b7e-4f27-b3f8-67e55dd03b01
# ╠═f8fa8b0a-f273-4009-8ec7-121ee213937f
# ╠═24efc612-3b74-404c-bbd3-31b22d655f56
# ╠═e0f1085e-6791-4e85-9747-6944c840cc85
# ╠═41f31996-b5ac-4500-a331-df691e54be59
# ╠═fe164335-c328-45c7-af42-8d05c515639a
# ╠═58163253-2c77-450e-8a08-901a25159aa7
# ╠═af068abe-6a23-4d1d-adf1-daca8fa567ff
# ╠═c38c520d-cfd3-4b1d-865a-38b71c244204
# ╠═b557f31f-d72e-4e3f-97e7-cbb21413d57c
# ╠═75079761-d1c2-48df-aa6d-531bf66186f5
# ╠═2039f019-7ffc-4c6a-ae33-05e2c24fc54b
# ╠═99673b9d-819e-4e32-ac36-80bdae1aa888
# ╠═e00cd98e-8490-4af4-aca1-402c3ae53f79
# ╠═e299e1ea-a00a-4dc4-8e36-cb91a254b5e9
# ╠═bf46e435-f483-448c-a6a1-6011fc5a78af
# ╠═ab253287-e6e7-4289-8a3c-32b09ee485ad
# ╠═b393139a-b734-441e-a6e8-f0f8fe68b33c
# ╠═71db0b3d-6af8-4019-a31f-12c532a3bc43
# ╠═3660b185-45b9-4bd4-9007-4d7a8014bd51
# ╠═3bc95456-b4b3-4d62-b022-5e72b8be9154
# ╠═e41a9ff9-0001-49d2-9f89-b60e42e74e81
# ╠═fed4e2b5-a962-4a69-975a-8ae80296c523
# ╠═b2332c65-5054-4b5f-91b0-63abd18fdebd
# ╠═68d35ad6-a77e-4eda-af4f-c914dc69fd25
# ╠═e4571a77-7358-4b0c-959b-0540ae5a0dec
# ╠═e60a7f31-bbf4-4fc7-9ec8-c247036bd303
# ╠═66aa977f-a8e9-447e-b755-67569e23b25f
# ╠═bdbc1b33-d018-4bca-b7dc-d6b6188c047b
# ╠═e427fe79-5d08-45fc-9fc2-9706449ee1e7
# ╠═26e94c39-ff53-41e1-898b-22e1ffa29cab
# ╠═9e585547-192e-427f-ba19-715077d8cf1b
# ╠═75fc4288-f433-40a7-bcd0-5bab7d14ed3b
# ╠═f7a6998f-bdb4-4c86-9883-97742826b18b
# ╠═c68a4744-2fa9-4b67-8ae7-0a10e1c2e078
# ╠═5102fa56-b8f2-4817-b10f-fc4a4fd3b645
# ╠═9ea3bbdf-70c6-414e-8bc7-a3f4709f0198
# ╠═bd13a315-a8f9-49b3-84cb-0b9245b0bca1
# ╠═9dffdc1c-38d3-4b21-adb8-4ec7e7ea3d62
# ╠═055905f6-0e27-4635-837e-d2dcfe9122d8
# ╠═5917870f-453f-42b8-94a2-f2e39ff01f0f
# ╠═904c5b0a-1ee1-46f0-aee1-867169ffb096
# ╠═acf51970-f888-4741-a4c0-c6c562a9db15
# ╠═52a05fd9-ce3b-4d8f-8ef4-05ba62e83a8b
# ╠═0ad816c2-55d9-4461-8ef9-fa7e959ea8e8
# ╠═1ae356ee-f287-41e0-bb92-47b54dbd36c5
# ╠═25aa9975-b444-4c2d-9121-7f34e01c577d
# ╠═5328ff94-d492-4fde-9401-2e1b54405345
# ╠═9b32c621-f694-47cf-a945-0efc3b3e47a4
# ╠═dece773e-1fe4-4c1d-9e4e-c1b32dad107b
# ╠═a300a6ed-9955-4f83-b262-62ce10a540ed
# ╠═b8314e1c-fda4-442b-8e79-b867dd8d2533
# ╠═4b4f0d7f-e977-4415-8f3f-4a1f78478845
# ╠═6d5d9692-25c3-42f2-a45f-336c90585789
# ╠═30921264-6a59-4d35-8b7d-3e47445588be
# ╠═8408a1b8-5899-4a1e-a1fa-9882600c446d
# ╠═4fc4e5f5-acb3-4fb4-bcc8-4486c674b266
# ╠═b8890dbb-5ed7-44d3-9f0a-bbc9e304a1e0
# ╠═6961d1b0-0f7e-49a1-aa7c-1e2cc93c37db
# ╠═8f1eed65-2ff2-4b42-a2f4-10c2ac359bd3
# ╠═9e80322b-6ced-48a5-8832-445e3902b4a7
# ╠═6d0acdd8-a20e-4522-b7f9-1c942600d336
# ╠═ef16799f-45d2-4dc2-8c85-9e89941f6d0c
# ╠═9bebfb65-3b2d-4441-91b7-c26293ccf901
# ╠═382cda6b-0181-4762-83cf-7e5a85eec1f3
# ╠═4a0820db-4ccb-4821-af26-d79782f8c7c5
# ╠═2f182a73-dcd3-4694-a028-f3102b74232b
