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

# ╔═╡ c017450a-f5a7-11ee-0164-85395fdbfd76
using DrWatson

# ╔═╡ 81dd640f-e0b5-4ef8-871b-6e83e5458fe4
@quickactivate

# ╔═╡ a9932afd-b2c3-4495-9918-5505b47eafa3
push!(LOAD_PATH,srcdir())

# ╔═╡ 4d45344c-a13d-476e-8470-f2be4ffb18e9
using Revise, gpu,highdim, MLJ,PlutoUI, SPDE

# ╔═╡ eeb12461-fd30-4544-9e9e-82ed1f3859b5
using DataFrames

# ╔═╡ 2ada319a-ae31-485d-97b1-a47d26712799
using JLD2

# ╔═╡ e10997ad-16c3-4fdb-b449-17dc24a56e4c
using CairoMakie, Makie, AlgebraOfGraphics

# ╔═╡ 07b9f4b4-76c7-43dd-8479-b107ed7abc4f
using Glob

# ╔═╡ db632c71-51a7-4564-b30f-69a73f12cfea
import Plots

# ╔═╡ 249e4694-001d-4b42-af3b-cdf9c14a1f14
begin
	σ(x) = 0.5x
	truth(x) = σ(x)^2
end

# ╔═╡ ab4cfa67-c15f-424f-8e05-ebf3ae70c922
for sample in 2:3
	#sol_temp = SPDE.generate_solution(σ)
	#name = savename(Dict("sigma"=>"sin(x)","sample"=>sample))
	#@save datadir(name*".jld2") sol_temp
end

# ╔═╡ eab99447-7db9-493c-b52b-a066e015c06a
#JLD2.@load datadir("solutions",readdir(datadir("solutions"))[1]) sol_temp

# ╔═╡ 940e9353-56f7-41af-862f-10fcfaa4a0ad
#sol = sol_temp

# ╔═╡ 29dd233d-01e0-4d15-88aa-e859201727a9
savename(Dict("w1"=>1.,"w2"=>2.))

# ╔═╡ e41727ff-363b-47c4-bb42-fcb38c2b750c
2^7

# ╔═╡ 24bd0085-8b15-4223-a588-edfcb0a6dc67
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

# ╔═╡ 7a259b32-8ea1-401b-ba89-ce7a719fd204
function filter_filenames(folder_path, epsilon, sigma, tquot, xquot)
    # Construct the pattern to match the files based on specified variables
    pattern = "tquot=$(tquot)_xquot=$(xquot).jld2"
    
    # Use Glob to find files matching the pattern
    matching_files = glob(pattern, folder_path)
    
    return matching_files
end

# ╔═╡ 1852b239-3634-4080-a1d7-3303d75239d4


# ╔═╡ a35d1717-e1c5-4826-a8ab-297cd10affd6
df_exp = collect_results(datadir("simulations"))

# ╔═╡ ed8e706e-f858-4be1-aec8-ded2d3c4953c
df_exp.new_dx = log.(df_exp.dx .* df_exp.xquot)

# ╔═╡ 544d230f-052f-4813-8eff-d080bedabf5a
df_exp.new_dt = df_exp.dt .* df_exp.tquot

# ╔═╡ 4ff556cc-4351-4127-851d-62193ac73237
data(df_exp)*mapping(:epsilon=>log,:l2=>log,color=:sigma,layout=:new_dx=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ c0701f1b-5231-4a00-b7c7-877bfaf2281b
df_exp

# ╔═╡ 628d5246-63a6-4ddb-ab26-d932996908e0
xs,ts = size(sol)

# ╔═╡ 2b795a71-a92f-468a-8614-2caaa55209c3
sol[xs ÷ 2,end]

# ╔═╡ 4b05e150-3423-4089-be24-769d9601800f


# ╔═╡ f1cf062f-2984-4273-9fa9-6b4374af30a0
begin
	L=1 # right boundary of space
	tmax = 0.3;
	#σ = x-> 0.5*sin(2pi*x/6)
	h=2^10
	nx = h;
	nt= 2^20
	dx = L/(nx-1); 
	dt = tmax/(nt-1);
	tmax = 0.3; 
end

# ╔═╡ 1395c711-b758-4980-840b-533e95d82973
begin
	param1 = Dict("epsilon"=>[2^i*dx for i in 1:5],
	"xquot" => 2^0,
	"tquot"=>2^0,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
	param2 = Dict("epsilon"=>[2^i*dx for i in 1:6],
	"xquot" => 2^1,
	"tquot"=>2^2,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
	param3 = Dict("epsilon"=>[2^i*dx for i in 2:7],
	"xquot" => 2^2,
	"tquot"=>2^4,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
	param4 = Dict("epsilon"=>[2^i*dx for i in 3:7],
	"xquot" => 2^3,
	"tquot"=>2^6,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
	param5 = Dict("epsilon"=>[2^i*dx for i in 4:7],
	"xquot" => 2^4,
	"tquot"=>2^8,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
	param6 = Dict("epsilon"=>[2^i*dx for i in 5:7],
	"xquot" => 2^5,
	"tquot"=>2^10,
	"sigma" => "0.5x",
	"dx"=>dx,
	"dt"=>dt)
end

# ╔═╡ 7076d8a1-7eb2-4ed0-b86d-0bf851285c2c
long_params = vcat(dict_list.([param2,param3,param4,param5,param6])...)

# ╔═╡ 69bd2991-dfb7-47c6-9ec3-0bfeb282321e
for sample in 1:6
	σ(x) = 0.5x
	sol = SPDE.generate_solution(σ)
	for (i, d) in enumerate(long_params)
	    mach,f = SPDE.paper_exp(sol,d,σ)
		d["sample"] = sample
		println(d)
		wsave(datadir("simulations", savename(d, "jld2")), f)
		MLJ.save(datadir("machines",savename(d,"jld2")),mach)
	end
end


# ╔═╡ cbd67162-38d1-4e6e-8a2e-98a3195c5a90


# ╔═╡ b5d60059-398b-42d8-946e-7fd5aab10605
size(sol)

# ╔═╡ 6b20b20d-caee-4923-9182-085fff1d11a7
df = SPDE.partial_integration(sol,dt,dx,2^5,2^10,128*dx)

# ╔═╡ 12f22559-abe2-4d6d-9022-20e793750ba8
begin
	using Statistics
	
	# Calculate the 5th and 95th percentiles
	lower_bound = quantile(df.x, 0.05)
	upper_bound = quantile(df.x, 0.95)
	
	# Filter the array to keep only the middle 90%
	filtered_data = filter(x -> x >= lower_bound && x <= upper_bound, df.x)
	umax = maximum(filtered_data)
	umin = minimum(filtered_data)
end

# ╔═╡ bb0563be-a270-425d-95fd-4c530aafb01c


# ╔═╡ 3e99b0fd-e8ff-40f2-b408-82795ed4debd
minimum(df.x)

# ╔═╡ 6c259a07-aaff-4947-8b3a-58dfca6df403
histogram(df.x)

# ╔═╡ 53106e97-fbd0-4fb5-8cb1-026dde82c3c6
umax

# ╔═╡ 054e4ae9-ecbe-4afa-8478-6dbaf5e9634a
(2*dx ÷ (4*dt))

# ╔═╡ d0ab1075-b230-4309-b2d0-3968cc6e4f6b
scatter(df.x,df.y)

# ╔═╡ f9c41ad3-a9af-4057-8030-62926ad71087
mach = SPDE.train_tuned(df)

# ╔═╡ 560ef5d8-043a-4ed1-9a72-f90d74b3cf10
fun_est = SPDE.mach_to_func(mach)

# ╔═╡ 8b8e8047-0f07-4dea-947f-432cf4d5d1f2
begin
	plot(fun_est,xlims=(umin,umax))
	plot!(truth)
end

# ╔═╡ e4ed6b83-1423-49bd-ac3b-733d4bc0f57e
begin

		mail_funcs=filter_filenames(datadir("machines"), string(epsilons[i]), "sigma2", string(8), string(2)) .|> machine .|> mach_to_func
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

# ╔═╡ 4c21673c-1551-4643-b148-c77075a22716
# ╠═╡ disabled = true
#=╠═╡
#paths= datadir("machines/").*readdir(datadir("machines"))
  ╠═╡ =#

# ╔═╡ cc0c9bc5-b047-4ce9-be65-ca4ea528fe7c
#funcs = machine.(paths) .|> SPDE.mach_to_func

# ╔═╡ ed9756ee-f062-42de-8fa0-305e4b78061a
begin
	plt = plot()
	for fun in funcs
		plot!(fun,xlims=(0.5,5))
	end
	plt
end

# ╔═╡ c19bd304-4ee0-4eb5-82a7-966370de4b9b
pattern = "*0.5x*tquot=16_xquot=4.jld2"

# ╔═╡ 2d35be40-2e54-4b67-9c2a-bc7ac632bcc5
glob

# ╔═╡ ecea9239-4ead-4063-8c12-1ee2511c58cf
ests=glob(pattern,datadir("machines")) .|> machine .|> SPDE.mach_to_func

# ╔═╡ 0837cb26-3a68-47b0-b5b8-70daae283d04
begin
	plt2 = Plots.plot()
	for est in ests
		Plots.plot!(est,xlims=(0.5,5))
	end
	Plots.plot!(truth)
	plt2
end

# ╔═╡ fab69841-6fb5-4cc2-bd57-88da4aa1baf9
paths = glob(pattern,datadir("machines"))

# ╔═╡ 1ce56563-5e96-4116-b69e-7373a960e98f
  df_s1 = filter(:sigma=> x->x=="0.5x",df_exp).l2

# ╔═╡ e53652fe-095e-42a8-8fdc-e31645b7bf1e


# ╔═╡ b1c343e5-753b-415f-8aaf-217cab06b1de
@bind idx PlutoUI.Slider(1:length(paths))

# ╔═╡ 871f578e-0548-4ff0-a179-53de82b44412
begin
	Plots.plot(truth)
	Plots.plot!(ests[idx],xlims=(0.5,6),label=paths[idx][end-55:end],ylims=(0,20))
end

# ╔═╡ 965e6a07-1b76-4726-ac75-8c334d684f6a
glob(pattern,datadir("machines"))[idx]

# ╔═╡ 05e4d89a-b751-4582-b428-312adbd0af5e
SPDE.get_all_losses( machine.(paths)[1],truth,(0.5,5),1000)

# ╔═╡ Cell order:
# ╠═c017450a-f5a7-11ee-0164-85395fdbfd76
# ╠═81dd640f-e0b5-4ef8-871b-6e83e5458fe4
# ╠═a9932afd-b2c3-4495-9918-5505b47eafa3
# ╠═4d45344c-a13d-476e-8470-f2be4ffb18e9
# ╠═db632c71-51a7-4564-b30f-69a73f12cfea
# ╠═eeb12461-fd30-4544-9e9e-82ed1f3859b5
# ╠═2ada319a-ae31-485d-97b1-a47d26712799
# ╠═249e4694-001d-4b42-af3b-cdf9c14a1f14
# ╠═ab4cfa67-c15f-424f-8e05-ebf3ae70c922
# ╠═eab99447-7db9-493c-b52b-a066e015c06a
# ╠═940e9353-56f7-41af-862f-10fcfaa4a0ad
# ╠═29dd233d-01e0-4d15-88aa-e859201727a9
# ╠═1395c711-b758-4980-840b-533e95d82973
# ╠═e41727ff-363b-47c4-bb42-fcb38c2b750c
# ╠═7076d8a1-7eb2-4ed0-b86d-0bf851285c2c
# ╠═69bd2991-dfb7-47c6-9ec3-0bfeb282321e
# ╠═24bd0085-8b15-4223-a588-edfcb0a6dc67
# ╠═7a259b32-8ea1-401b-ba89-ce7a719fd204
# ╠═1852b239-3634-4080-a1d7-3303d75239d4
# ╠═a35d1717-e1c5-4826-a8ab-297cd10affd6
# ╠═ed8e706e-f858-4be1-aec8-ded2d3c4953c
# ╠═544d230f-052f-4813-8eff-d080bedabf5a
# ╠═4ff556cc-4351-4127-851d-62193ac73237
# ╠═e10997ad-16c3-4fdb-b449-17dc24a56e4c
# ╠═c0701f1b-5231-4a00-b7c7-877bfaf2281b
# ╠═628d5246-63a6-4ddb-ab26-d932996908e0
# ╠═2b795a71-a92f-468a-8614-2caaa55209c3
# ╠═4b05e150-3423-4089-be24-769d9601800f
# ╠═f1cf062f-2984-4273-9fa9-6b4374af30a0
# ╠═cbd67162-38d1-4e6e-8a2e-98a3195c5a90
# ╠═b5d60059-398b-42d8-946e-7fd5aab10605
# ╠═6b20b20d-caee-4923-9182-085fff1d11a7
# ╠═bb0563be-a270-425d-95fd-4c530aafb01c
# ╠═3e99b0fd-e8ff-40f2-b408-82795ed4debd
# ╠═6c259a07-aaff-4947-8b3a-58dfca6df403
# ╠═12f22559-abe2-4d6d-9022-20e793750ba8
# ╠═53106e97-fbd0-4fb5-8cb1-026dde82c3c6
# ╠═054e4ae9-ecbe-4afa-8478-6dbaf5e9634a
# ╠═d0ab1075-b230-4309-b2d0-3968cc6e4f6b
# ╠═f9c41ad3-a9af-4057-8030-62926ad71087
# ╠═560ef5d8-043a-4ed1-9a72-f90d74b3cf10
# ╠═8b8e8047-0f07-4dea-947f-432cf4d5d1f2
# ╠═e4ed6b83-1423-49bd-ac3b-733d4bc0f57e
# ╠═4c21673c-1551-4643-b148-c77075a22716
# ╠═cc0c9bc5-b047-4ce9-be65-ca4ea528fe7c
# ╠═ed9756ee-f062-42de-8fa0-305e4b78061a
# ╠═07b9f4b4-76c7-43dd-8479-b107ed7abc4f
# ╠═c19bd304-4ee0-4eb5-82a7-966370de4b9b
# ╠═2d35be40-2e54-4b67-9c2a-bc7ac632bcc5
# ╠═ecea9239-4ead-4063-8c12-1ee2511c58cf
# ╠═0837cb26-3a68-47b0-b5b8-70daae283d04
# ╠═871f578e-0548-4ff0-a179-53de82b44412
# ╠═fab69841-6fb5-4cc2-bd57-88da4aa1baf9
# ╠═1ce56563-5e96-4116-b69e-7373a960e98f
# ╠═e53652fe-095e-42a8-8fdc-e31645b7bf1e
# ╠═b1c343e5-753b-415f-8aaf-217cab06b1de
# ╠═965e6a07-1b76-4726-ac75-8c334d684f6a
# ╠═05e4d89a-b751-4582-b428-312adbd0af5e
