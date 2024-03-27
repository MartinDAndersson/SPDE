### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 40df38b6-01ab-43af-95c0-5172268402bf
begin
	import Pkg
	using Revise
	#Pkg.activate(".")
	using LinearAlgebra, Random
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

# ╔═╡ 8afca024-c17d-490e-b740-d71ca7030c60
using DrWatson

# ╔═╡ 1fbc96ca-2004-4ce2-b50a-c4007742b9d3
@quickactivate "SPDE"

# ╔═╡ 1b7dfebe-5766-4568-b713-0bed282fe338
using Glob

# ╔═╡ 7d1db358-227f-412f-844c-be57464a41fc
using AlgebraOfGraphics

# ╔═╡ f9000274-bcd1-490c-b5c6-217145eca452
using CairoMakie

# ╔═╡ 2de77140-e5f2-11ee-0e5b-03bf773664a6
function filter_filenames(folder_path, epsilon, sigma, tquot, xquot)
    # Construct the pattern to match the files based on specified variables
    pattern = "epsilon=$(epsilon)_sample=*_sigma=$(sigma)_tquot=$(tquot)_xquot=$(xquot).jld2"
    
    # Use Glob to find files matching the pattern
    matching_files = glob(pattern, folder_path)
    
    return matching_files
end


# ╔═╡ 2462eeb6-7c99-4f7f-805b-89c8b72f3da4
epsilons=[0.125,0.0626,0.0313,0.0157,0.00783,0.00196]

# ╔═╡ 072bbbb3-a134-478e-b74f-dbcdb1e9e948
path_to_files = datadir("machines")

# ╔═╡ ff252ca5-6976-4d4d-a9f7-9eba8f2bbc54
path_mail = datadir("mail_machines")

# ╔═╡ 6966ac5d-8304-42c4-92a7-9fcd7d5a1e17
function mach_to_func(mach)
	est_wrapper(x) = MLJ.predict(mach,DataFrame(:x=>[x]))[1]
	return est_wrapper
end

# ╔═╡ 2e50b67e-374e-4318-98e6-04724557006b
begin
	σ(x) = 0.5x
	truth(x) = σ(x)^2
end

# ╔═╡ 0ad708e7-ecf1-43d3-81eb-818e9ec07217
begin
	path_samples = filter_filenames(path_to_files,string(eps),"sigma1",string(4),string(2))
	fun_samples = machine.(path_samples) .|> mach_to_func
end

# ╔═╡ 07f95b8e-7c48-421f-8474-4cf8cd35636b
begin
	df_quot = DataFrame(:eps=>Float64[],:tq=>Float64[],:xq=>Float64[],:quot=>Float64[],
	:std_est=>Float64[])
	for epsilon in epsilons
		#sample_points=mean_quot.(collect(2:0.01:6)) #|> mean
		Threads.@threads for tq in [2^i for i in 0:7]
			for xq in [2^i for i in 0:5]
				path_samples = filter_filenames(path_to_files,string(epsilon),"sigma2",string(tq),string(xq))
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

# ╔═╡ 8821eb42-5546-4924-ae73-8ef0efad7ef0


# ╔═╡ 24a5dbc1-ef5a-4e97-b7cf-5be7b4f175f5
data(df_quot)*mapping(:tq=>log,:quot=>log,layout=:xq=>nonnumeric,color=:eps=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ 78be669f-a5f2-4200-9adf-5c48690df4d9
begin
		L=1; 
		tmax = 0.3;
		h=2^9
		nx = h;
		nt= 2^18
		dx = L/(nx-1); 
		dt = tmax/(nt-1); 
end

# ╔═╡ 6f687d9c-ee1f-4316-a989-5997f0720e7b
2*dx

# ╔═╡ 81455346-6380-4eef-abed-be7f02fcd954
dx

# ╔═╡ 1ca32b65-b831-4a71-8098-25d13010ac02
64*dt

# ╔═╡ 439ddef4-c94d-46f1-a8d8-ff33951dfd4b
2^6*dt

# ╔═╡ 82f3f81a-0102-4cd9-baeb-87ec34e19627
dx

# ╔═╡ 7b8b806b-8061-4f3f-a601-c195b751485c


# ╔═╡ 146ca8a2-ddfb-4dd4-b4d9-8b6b44634329


# ╔═╡ a62031a0-1512-4664-874c-a04c41b14140


# ╔═╡ a7f5945d-3f97-4e45-b8fc-2793da6cef78


# ╔═╡ 945effb0-0623-48c5-98b5-9c42e085af50


# ╔═╡ 1e241fe9-3b2f-4476-a84b-4f7fbafbd411


# ╔═╡ c407de3e-3b43-4372-bcc7-ff0cbd77d59c


# ╔═╡ 35a01076-7f42-4cc7-a668-8b81b22ecd44


# ╔═╡ f84574fd-a1f6-4fc7-87f3-12770ec8faa6
test_array = vcat(1:0.1:2,3.5:0.01:6)

# ╔═╡ 1fc70c91-aa93-4313-9082-e2ea5357eae0
begin
		anim=Plots.@animate for i in [2,3,4,5]
			eps = epsilons[i]
			mail_funcs=filter_filenames(datadir("mail_machines"), string(eps), "sigma3", string(4), string(2)) .|> machine .|> mach_to_func
			
			mean_mail(x) = [fun(x) for fun in mail_funcs] |> mean

			
			mail_truth(x) = (0.5*sin(6*x/(2*pi)))^2
			mean_quot(x) = mean_mail(x)/mail_truth(x)
			quot_array=mean_quot.(test_array)#|> mean
			quot_est = mean(quot_array)
			std_est = std(quot_array)
			adjusted_estimate(x) = mean_mail(x)/quot_est
		
			plots_samples = Plots.plot()
			for fun in mail_funcs
				Plots.plot!(fun,xlims=(1,6),label="",alpha=0.5)
			end
			Plots.plot!(mail_truth,label=L"σ^2",color=:black,linewidth=2.)
			Plots.plot!(adjusted_estimate,label="adjusted regression",linewidth=2.,color=:blue)
			#plots_samples

			Plots.plot!(mean_mail,label="regression mean",linewidth=2,color=:red,legend=:topright,ylims=(0,0.6),xlims=(1,6),title="ϵpsilon=$eps")
		end
end

# ╔═╡ f39aad24-43c6-4c97-860f-330f578a8250


# ╔═╡ a2baa62a-89cc-48fc-8d09-a0e2eb6c440b
Plots.gif(anim,fps=1)

# ╔═╡ ef965c9d-d1b6-473a-aa13-5aa838052edf


# ╔═╡ c9a4c6f2-e01e-42d2-9679-af1f2ff74f29


# ╔═╡ a97a114a-16ce-4411-b162-75d6435e28dc


# ╔═╡ Cell order:
# ╠═8afca024-c17d-490e-b740-d71ca7030c60
# ╠═1fbc96ca-2004-4ce2-b50a-c4007742b9d3
# ╠═1b7dfebe-5766-4568-b713-0bed282fe338
# ╠═40df38b6-01ab-43af-95c0-5172268402bf
# ╠═7d1db358-227f-412f-844c-be57464a41fc
# ╠═2de77140-e5f2-11ee-0e5b-03bf773664a6
# ╠═2462eeb6-7c99-4f7f-805b-89c8b72f3da4
# ╠═072bbbb3-a134-478e-b74f-dbcdb1e9e948
# ╠═ff252ca5-6976-4d4d-a9f7-9eba8f2bbc54
# ╠═6966ac5d-8304-42c4-92a7-9fcd7d5a1e17
# ╠═2e50b67e-374e-4318-98e6-04724557006b
# ╠═0ad708e7-ecf1-43d3-81eb-818e9ec07217
# ╠═07f95b8e-7c48-421f-8474-4cf8cd35636b
# ╠═8821eb42-5546-4924-ae73-8ef0efad7ef0
# ╠═f9000274-bcd1-490c-b5c6-217145eca452
# ╠═6f687d9c-ee1f-4316-a989-5997f0720e7b
# ╠═24a5dbc1-ef5a-4e97-b7cf-5be7b4f175f5
# ╠═78be669f-a5f2-4200-9adf-5c48690df4d9
# ╠═81455346-6380-4eef-abed-be7f02fcd954
# ╠═1ca32b65-b831-4a71-8098-25d13010ac02
# ╠═439ddef4-c94d-46f1-a8d8-ff33951dfd4b
# ╠═82f3f81a-0102-4cd9-baeb-87ec34e19627
# ╠═7b8b806b-8061-4f3f-a601-c195b751485c
# ╠═1fc70c91-aa93-4313-9082-e2ea5357eae0
# ╠═146ca8a2-ddfb-4dd4-b4d9-8b6b44634329
# ╠═a62031a0-1512-4664-874c-a04c41b14140
# ╠═a7f5945d-3f97-4e45-b8fc-2793da6cef78
# ╠═945effb0-0623-48c5-98b5-9c42e085af50
# ╠═1e241fe9-3b2f-4476-a84b-4f7fbafbd411
# ╠═c407de3e-3b43-4372-bcc7-ff0cbd77d59c
# ╠═35a01076-7f42-4cc7-a668-8b81b22ecd44
# ╠═f84574fd-a1f6-4fc7-87f3-12770ec8faa6
# ╠═f39aad24-43c6-4c97-860f-330f578a8250
# ╠═a2baa62a-89cc-48fc-8d09-a0e2eb6c440b
# ╠═ef965c9d-d1b6-473a-aa13-5aa838052edf
# ╠═c9a4c6f2-e01e-42d2-9679-af1f2ff74f29
# ╠═a97a114a-16ce-4411-b162-75d6435e28dc
