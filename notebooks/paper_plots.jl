### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 695bfa62-da5a-4f72-a975-edabb5aabfb0
begin
	using Pkg
	Pkg.add("DrWatson")
end

# ╔═╡ c017450a-f5a7-11ee-0164-85395fdbfd76
using DrWatson

# ╔═╡ 81dd640f-e0b5-4ef8-871b-6e83e5458fe4
@quickactivate

# ╔═╡ a9932afd-b2c3-4495-9918-5505b47eafa3
push!(LOAD_PATH,srcdir())

# ╔═╡ 4d45344c-a13d-476e-8470-f2be4ffb18e9
using Revise, gpu,highdim, MLJ,PlutoUI, SPDE

# ╔═╡ e10997ad-16c3-4fdb-b449-17dc24a56e4c
using CairoMakie, Makie, AlgebraOfGraphics

# ╔═╡ fab50513-81e6-463f-8c5b-516db17ab5b4
using Distributions

# ╔═╡ eeb12461-fd30-4544-9e9e-82ed1f3859b5
using DataFrames

# ╔═╡ 2ada319a-ae31-485d-97b1-a47d26712799
using JLD2

# ╔═╡ 07b9f4b4-76c7-43dd-8479-b107ed7abc4f
using Glob

# ╔═╡ 3f6e19b5-3aa3-4be8-8f5a-6e63b2219d00
using DifferentialEquations.EnsembleAnalysis

# ╔═╡ 52501b42-4601-4df5-be66-d7ebd56e7404
using DifferentialEquations

# ╔═╡ b6c46a6c-4c57-4bc7-8064-86051f152f60
using Statistics

# ╔═╡ 7ea44d8c-15f3-42c3-acea-36f0f2c1bc71
using Trapz

# ╔═╡ db632c71-51a7-4564-b30f-69a73f12cfea
import Plots

# ╔═╡ 249e4694-001d-4b42-af3b-cdf9c14a1f14
begin
	#σ(x) = 0.5x
	
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

# ╔═╡ 9dafd043-d524-4caa-8137-2e3855ba47de
Dict("sigma1"=>x->0.5x,"sigma2"=>x->sin(x),"sigma3"=>x->sin(2x),"sigma4"=>x->0.5x+sin(x))

# ╔═╡ db7d37fe-ea55-4a00-a559-e080414e6f40
begin
	sigma = "sigma2"
	σ(x) = 1.
	truth(x) = σ(x)^2
	#samples=1
end

# ╔═╡ f1cf062f-2984-4273-9fa9-6b4374af30a0
begin
	L=1 # right boundary of space
	tmax = 0.3;
	#σ = x-> 0.5*sin(2pi*x/6)
	h=2^8
	nx = h;
	nt= 2^18
	dx = L/(nx-1); 
	dt = tmax/(nt-1);
	tmax = 0.3; 
end

# ╔═╡ a649f9d6-544a-4d36-ac6c-4b550c1b884c
w = Dict("epsilon"=>[2^i*dx for i in 1:5],
	"xquot" => 2^0,
	"tquot"=>2^0,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)

# ╔═╡ 1395c711-b758-4980-840b-533e95d82973
begin
	param1 = Dict("epsilon"=>[i*dx for i in 11:30],
	"xquot" => 2^0,
	"tquot"=>2^0,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
	param2 = Dict("epsilon"=>[i*dx for i in 6:1:20],
	"xquot" => 2^1,
	"tquot"=>2^2,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
	param3 = Dict("epsilon"=>[i*dx for i in 20:1:40],
	"xquot" => 2^2,
	"tquot"=>2^4,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
	param4 = Dict("epsilon"=>[2^i*dx for i in 0:6],
	"xquot" => 2^3,
	"tquot"=>2^6,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
	param5 = Dict("epsilon"=>[2^i*dx for i in 4:6],
	"xquot" => 2^4,
	"tquot"=>2^8,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
	param6 = Dict("epsilon"=>[2^i*dx for i in 5:6],
	"xquot" => 2^5,
	"tquot"=>2^10,
	"sigma" => sigma,
	"dx"=>dx,
	"dt"=>dt)
end

# ╔═╡ e41727ff-363b-47c4-bb42-fcb38c2b750c
2^7

# ╔═╡ 7076d8a1-7eb2-4ed0-b86d-0bf851285c2c
long_params = vcat(dict_list.([param1,param2,param3,param4,param5,param6])...)

# ╔═╡ 69bd2991-dfb7-47c6-9ec3-0bfeb282321e
for sample in 1:1
	
	sol = SPDE.generate_solution(σ)
	for (i, d) in enumerate(dict_list(param1))
	    mach,f = SPDE.paper_exp(sol,d,σ)
		d["sample"] = sample
		println(d)
		wsave(datadir("fel/simulations", savename(d, "jld2")), f)
		MLJ.save(datadir("fel/machines",savename(d,"jld2")),mach)
	end
end


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

# ╔═╡ a35d1717-e1c5-4826-a8ab-297cd10affd6
begin
	df_exp = collect_results(datadir("simulations")) |> dropmissing!
	df_exp.new_dx = df_exp.dx .* df_exp.xquot
	df_exp.new_dt = df_exp.dt .* df_exp.tquot
	df_exp.log_eps = log.(df_exp.epsilon)
	df_exp.log_dx = log.(df_exp.new_dx)
	#df_exp = filter(row -> (row.sigma in ["sin(x)", sigma]), df_exp)
end

# ╔═╡ 195844df-031e-46e6-84ab-e9d0544dd24b
begin
	group_columns = ["tquot","xquot","log_eps","sigma","log_dx"]
	grouped_df = groupby(df_exp,group_columns)
	mean_df = combine(grouped_df, [:l1 => mean => :l1_mean,:l2=>mean=>:l2_mean])
end

# ╔═╡ ae1f1065-ba42-4059-961a-3790f1690099
begin
	group_again = ["tquot","xquot","sigma","log_dx"]
	group2 = groupby(mean_df,group_again)
	min_l1_mean_df = combine(group2) do subdf
    return subdf[argmin(subdf.l1_mean), :]
end
end

# ╔═╡ 29f0d683-a9ac-447a-ba3f-0d0472d3dae5
 collect_results(datadir("simulations")) |> dropmissing!

# ╔═╡ ee056144-06f4-4c62-bbcb-697f2284b2fc


# ╔═╡ b3d7186d-3957-4f0b-8812-ce5a7193289d


# ╔═╡ ed8e706e-f858-4be1-aec8-ded2d3c4953c
eps_func(dx) = dx^(0.82)

# ╔═╡ a1879e1a-4692-4b7c-94d5-9873e3973eb3
eps_func(dx)

# ╔═╡ 4ff556cc-4351-4127-851d-62193ac73237
data(df_exp)*mapping(:epsilon=>log,:l1=>log,color=:sigma,layout=:log_dx=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ 306ee4e7-4bd0-4ec9-8ecd-affcd1679d5a
data(df_exp)*mapping(:epsilon=>log,:l2=>log,color=:sigma,layout=:log_dx=>nonnumeric)*visual(Scatter) |> draw

# ╔═╡ b1e45910-eda1-4128-a76c-3ee42b1560c2
data(mean_df)*mapping(:log_eps,:l1_mean,color=:sigma,layout=:log_dx=>nonnumeric)*(visual(Scatter)+smooth())|> draw

# ╔═╡ 3d23d188-9087-4b3a-b4ce-d82b1449fb66
data(min_l1_mean_df)*mapping(:log_dx,:log_eps,layout=:sigma,color=:sigma)*(linear()+visual(Scatter) )|> draw

# ╔═╡ 76eeb2de-1c8c-413f-a068-9592344a5292
data(min_l1_mean_df)*mapping(:log_dx,:l2_mean=>x->log(sqrt(x)),layout=:sigma,color=:sigma)*(linear()+visual(Scatter)) |> draw

# ╔═╡ eac0bc9e-310c-494c-be30-7431fc6edcc6
mean(min_l1_mean_df.log_eps ./ min_l1_mean_df.log_dx )

# ╔═╡ ab509f29-8a18-453b-9f74-5558c15f7148
std(min_l1_mean_df.log_eps ./ min_l1_mean_df.log_dx )

# ╔═╡ f9e3271a-bce7-4115-8139-e05337e77141
df_exp

# ╔═╡ cbd67162-38d1-4e6e-8a2e-98a3195c5a90


# ╔═╡ b5d60059-398b-42d8-946e-7fd5aab10605
#size(sol)

# ╔═╡ 6b20b20d-caee-4923-9182-085fff1d11a7
df = SPDE.partial_integration(sol,dt,dx,2^5,2^10,128*dx)

# ╔═╡ bb0563be-a270-425d-95fd-4c530aafb01c


# ╔═╡ 3e99b0fd-e8ff-40f2-b408-82795ed4debd
#minimum(df.x)

# ╔═╡ 6c259a07-aaff-4947-8b3a-58dfca6df403
#histogram(df.x)

# ╔═╡ 12f22559-abe2-4d6d-9022-20e793750ba8
#=begin
	using Statistics
	
	# Calculate the 5th and 95th percentiles
	lower_bound = quantile(df.x, 0.05)
	upper_bound = quantile(df.x, 0.95)
	
	# Filter the array to keep only the middle 90%
	filtered_data = filter(x -> x >= lower_bound && x <= upper_bound, df.x)
	umax = maximum(filtered_data)
	umin = minimum(filtered_data)
end=#

# ╔═╡ 53106e97-fbd0-4fb5-8cb1-026dde82c3c6
#umax

# ╔═╡ 054e4ae9-ecbe-4afa-8478-6dbaf5e9634a
#(2*dx ÷ (4*dt))

# ╔═╡ d0ab1075-b230-4309-b2d0-3968cc6e4f6b
#scatter(df.x,df.y)

# ╔═╡ f9c41ad3-a9af-4057-8030-62926ad71087
#mach = SPDE.train_tuned(df)

# ╔═╡ 560ef5d8-043a-4ed1-9a72-f90d74b3cf10
#fun_est = SPDE.mach_to_func(mach)

# ╔═╡ 4c21673c-1551-4643-b148-c77075a22716
# ╠═╡ disabled = true
#=╠═╡
#paths= datadir("machines/").*readdir(datadir("machines"))
  ╠═╡ =#

# ╔═╡ cc0c9bc5-b047-4ce9-be65-ca4ea528fe7c
#funcs = machine.(paths) .|> SPDE.mach_to_func

# ╔═╡ 2d35be40-2e54-4b67-9c2a-bc7ac632bcc5
begin
	new_sol = SPDE.generate_solution(σ,nx,nt)
end

# ╔═╡ ab0d3e13-7256-402e-b344-98ea9948880b
SPDE.L_op(Array(new_sol),dx,dt,1,1)

# ╔═╡ 6cc4139d-1b2e-44c9-9bdb-db981c211730
new_sol.t[2]-new_sol.t[1]

# ╔═╡ 70343c71-5584-425e-aa68-caaa44e3c7bb
new_sol[1]

# ╔═╡ 122712f6-61f2-4288-8194-e56d33298059
sum=EnsembleSummary(new_sol)

# ╔═╡ a928ccfd-74d7-4b4b-9617-707b736bb0e1
begin
	#df=SPDE.partial_integration(new_sol,dt,dx,1,1,4dx)
end

# ╔═╡ 61020487-999d-4af7-b3f5-8fb6ef5258b7
Plots.plot(sum,idxs=[100])

# ╔═╡ e7e4d71e-45f0-4d6c-a973-5e46916ce52c
Array(new_sol)[5,1000,:]

# ╔═╡ 99a423ca-5966-48c0-8828-93368e983ed0


# ╔═╡ d5378cc9-382a-45ba-aabb-0d6c5269c71a
new_sol[1].t[end]

# ╔═╡ 84ad3671-4d5c-48e0-a459-1cf6414051e6
mach=SPDE.train_tuned(df)

# ╔═╡ 7da60210-a0b9-4556-95e7-86e429c0a3de
train_func = SPDE.mach_to_func(mach)

# ╔═╡ 7150257c-5bb1-4695-8dd6-829fc7557732
mean(df.y)

# ╔═╡ a1672a38-9a19-4dec-9b5b-38f053c24382
sqrt(2*4*dt*dx)

# ╔═╡ 26317fb6-4c61-4ae2-83c8-de8383d463ce
new_sol.t |> length

# ╔═╡ de7ba506-26c8-4c55-8806-66c168ecd0a9
1/(dx*dt) |> sqrt

# ╔═╡ 1dd0676f-0e1b-415a-8209-9b6b34e304b6
begin
	Plots.plot(x->train_func(x),xlims=(0,6))
	Plots.plot!(truth)
end

# ╔═╡ c115fdc0-3940-470a-ba3e-b0d8d70583b6
new_sol.t[2]-new_sol.t[1]

# ╔═╡ e8314029-ac3a-4a73-b29d-e033f47f2886


# ╔═╡ e392b03c-835d-425c-b777-d3d31184e089
dt

# ╔═╡ 123fe936-9fb2-4084-94b5-29323e2fbace
plot(df.x,df.y,alpha=0.02)

# ╔═╡ 90e9b13d-5172-4b8d-91c1-0f2848d1b1e0
X=SPDE.L_op(Array(new_sol),2*dt,1*dx,1,2)  

# ╔═╡ f4c87141-8029-4b38-b761-c4a8b9c1c0ba
heatmap(X[:,end-1000:end])

# ╔═╡ 917de72d-7592-41e9-a3ec-304161d6c629
dx

# ╔═╡ 523f9965-8533-4a76-9a38-97e8ba55e084


# ╔═╡ 4dcfed74-9fb3-4fbf-991f-777026a8bda2
(rand(X[2:end-1,2:end-2][:],10000) .* dt .* 1/dx) |> var

# ╔═╡ d602790f-f9bb-4388-b1e5-094127ca4eb0
scatter(X[128,end-10000:end].^2 .* dt,alpha=0.1)

# ╔═╡ 4ba1dc45-984c-465f-a592-890a68dd99f2
begin
	lower_bound = quantile(df.x, 0.05)
	upper_bound = quantile(df.x, 0.95)
	# Filter the array to keep only the middle 90%
	filtered_data = filter(row -> lower_bound < row.x < upper_bound, df)
	lower_bound = quantile(df.y, 0.4)
	upper_bound = quantile(df.y, 0.99)
	filtered_data =filter(row -> lower_bound < row.y , df)

	#umax = maximum(filtered_data)
	#umin = minimum(filtered_data)
end

# ╔═╡ 27e77ef4-7596-4d91-b5b5-86c1044a2b67
plot(filtered_data.x,filtered_data.y,alpha=0.01)

# ╔═╡ 43a4c5e5-140c-45ea-9b27-859f953aebb6
lower_bound

# ╔═╡ 9f4e342c-8973-4faa-81d8-16613cba858d
fit(Normal,df.y)

# ╔═╡ e3cd63e7-5b8d-4b37-b741-cb03806800e5
begin
	plt=Makie.density(df.y)
	Makie.plot!(fit(Normal,df.y[end-1000:end]),color=:red)
	plt
end

# ╔═╡ 36f74429-156f-4d68-85af-dfa8f90d9f1c
scatter(df.x,df.y,alpha=0.01)

# ╔═╡ c19bd304-4ee0-4eb5-82a7-966370de4b9b
pattern = "*"*sigma*"*tquot=4_xquot=2.jld2"

# ╔═╡ 453cd265-5617-4db4-b8fc-5a53f2d4fb59
2^8

# ╔═╡ ecea9239-4ead-4063-8c12-1ee2511c58cf
ests=glob(pattern,datadir("fel/machines")) .|> machine .|> SPDE.mach_to_func

# ╔═╡ 0837cb26-3a68-47b0-b5b8-70daae283d04
begin
	plt2 = Plots.plot()
	for est in ests
		Plots.plot!(x->est(x),xlims=(0.5,3),label="",alpha=0.5)
	end
	#Plots.plot!(truth)
	plt2
end

# ╔═╡ f22ed1bd-1ea9-4280-8bbf-dbf1c6f3d90b
mean_fun = (glob("*0.00782*"*sigma*"*",datadir("machines")) .|> machine .|> SPDE.mach_to_func)

# ╔═╡ 7517cd1d-4733-4829-aaa6-3c8a2e20b8e8
4*dx

# ╔═╡ e54c6768-fcfc-411b-97a4-927bb8e00a2c
A=rand(3,3)

# ╔═╡ 18f971be-5a24-47ed-bced-b7370ec2da58
range(0,2*3,length=100)

# ╔═╡ 9020b6ac-f1b6-40d3-b105-7dd6a62847e0
size(A)

# ╔═╡ 5413c0f0-6648-4676-a105-0527cbf66ad2
ests

# ╔═╡ fab69841-6fb5-4cc2-bd57-88da4aa1baf9
paths = glob(pattern,datadir("fel/machines"))

# ╔═╡ 23b26789-3d64-4165-8198-5513362495e7
length(paths)

# ╔═╡ b1c343e5-753b-415f-8aaf-217cab06b1de
@bind idx PlutoUI.Slider(1:length(paths))

# ╔═╡ 871f578e-0548-4ff0-a179-53de82b44412
begin
	Plots.plot(truth)
	Plots.plot!(x->ests[idx](x),xlims=(0.,3.),label=paths[idx][end-55:end],ylims=(0,1))
end

# ╔═╡ 4adb7af1-0cd1-452a-a1d1-c434bb494e2e
ests[idx].(0:0.01:5)./truth.(0:0.01:5) |> median

# ╔═╡ 1c760b9e-2b60-4ee0-a93d-d1eecb48f6bb
median(ests[idx].(0.5:0.01:5)./truth.(0.5:0.01:5))

# ╔═╡ 1ce56563-5e96-4116-b69e-7373a960e98f
  df_s1 = filter(:sigma=> x->x=="0.5x",df_exp).l2

# ╔═╡ e53652fe-095e-42a8-8fdc-e31645b7bf1e


# ╔═╡ 7a2da99c-f713-4fda-b33f-40433cef59ee
eps_func(2*dx)

# ╔═╡ 965e6a07-1b76-4726-ac75-8c334d684f6a
glob(pattern,datadir("machines"))[idx]

# ╔═╡ 05e4d89a-b751-4582-b428-312adbd0af5e
#SPDE.get_all_losses( machine.(paths)[1],truth,(0.5,5),1000)

# ╔═╡ af4d8058-5fd3-4f91-ad9b-a8db7ffc3b44
# ╠═╡ disabled = true
#=╠═╡
A = rand(10,10)
  ╠═╡ =#

# ╔═╡ f899052b-fffe-49df-9e74-11994b30083f
xs,ts = size(A)

# ╔═╡ cd9af91e-bc8f-4559-8d5d-6938c2606ae8
l1,l2 = (dx*xs,dt*ts)

# ╔═╡ 6b92f6f6-b575-4da9-9269-dac9e7260e4e
begin
	xran=range(0,l1,length=xs)
	yran=range(0,l2,length=ts)
end

# ╔═╡ 79a51857-5e3f-4f90-845a-50b1f35abc8e


# ╔═╡ 982dbd22-28f6-4b2d-84da-36b4305e3f8e
trapz((xran,yran),A)

# ╔═╡ 3e06f4da-0c9d-42b8-ba02-c9726aaa581c
trapz((1:xs,1:ts),dt*dx*A)

# ╔═╡ Cell order:
# ╠═695bfa62-da5a-4f72-a975-edabb5aabfb0
# ╠═c017450a-f5a7-11ee-0164-85395fdbfd76
# ╠═81dd640f-e0b5-4ef8-871b-6e83e5458fe4
# ╠═a9932afd-b2c3-4495-9918-5505b47eafa3
# ╠═4d45344c-a13d-476e-8470-f2be4ffb18e9
# ╠═e10997ad-16c3-4fdb-b449-17dc24a56e4c
# ╠═db632c71-51a7-4564-b30f-69a73f12cfea
# ╠═fab50513-81e6-463f-8c5b-516db17ab5b4
# ╠═eeb12461-fd30-4544-9e9e-82ed1f3859b5
# ╠═2ada319a-ae31-485d-97b1-a47d26712799
# ╠═249e4694-001d-4b42-af3b-cdf9c14a1f14
# ╠═ab4cfa67-c15f-424f-8e05-ebf3ae70c922
# ╠═eab99447-7db9-493c-b52b-a066e015c06a
# ╠═940e9353-56f7-41af-862f-10fcfaa4a0ad
# ╠═29dd233d-01e0-4d15-88aa-e859201727a9
# ╠═9dafd043-d524-4caa-8137-2e3855ba47de
# ╠═db7d37fe-ea55-4a00-a559-e080414e6f40
# ╠═f1cf062f-2984-4273-9fa9-6b4374af30a0
# ╠═a649f9d6-544a-4d36-ac6c-4b550c1b884c
# ╠═1395c711-b758-4980-840b-533e95d82973
# ╠═e41727ff-363b-47c4-bb42-fcb38c2b750c
# ╠═7076d8a1-7eb2-4ed0-b86d-0bf851285c2c
# ╠═69bd2991-dfb7-47c6-9ec3-0bfeb282321e
# ╠═24bd0085-8b15-4223-a588-edfcb0a6dc67
# ╠═7a259b32-8ea1-401b-ba89-ce7a719fd204
# ╠═195844df-031e-46e6-84ab-e9d0544dd24b
# ╠═ae1f1065-ba42-4059-961a-3790f1690099
# ╠═a35d1717-e1c5-4826-a8ab-297cd10affd6
# ╠═29f0d683-a9ac-447a-ba3f-0d0472d3dae5
# ╠═ab0d3e13-7256-402e-b344-98ea9948880b
# ╠═6cc4139d-1b2e-44c9-9bdb-db981c211730
# ╠═70343c71-5584-425e-aa68-caaa44e3c7bb
# ╠═ee056144-06f4-4c62-bbcb-697f2284b2fc
# ╠═b3d7186d-3957-4f0b-8812-ce5a7193289d
# ╠═ed8e706e-f858-4be1-aec8-ded2d3c4953c
# ╠═a1879e1a-4692-4b7c-94d5-9873e3973eb3
# ╠═4ff556cc-4351-4127-851d-62193ac73237
# ╠═306ee4e7-4bd0-4ec9-8ecd-affcd1679d5a
# ╠═b1e45910-eda1-4128-a76c-3ee42b1560c2
# ╠═3d23d188-9087-4b3a-b4ce-d82b1449fb66
# ╠═76eeb2de-1c8c-413f-a068-9592344a5292
# ╠═eac0bc9e-310c-494c-be30-7431fc6edcc6
# ╠═ab509f29-8a18-453b-9f74-5558c15f7148
# ╠═f9e3271a-bce7-4115-8139-e05337e77141
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
# ╠═4c21673c-1551-4643-b148-c77075a22716
# ╠═cc0c9bc5-b047-4ce9-be65-ca4ea528fe7c
# ╠═07b9f4b4-76c7-43dd-8479-b107ed7abc4f
# ╠═2d35be40-2e54-4b67-9c2a-bc7ac632bcc5
# ╠═3f6e19b5-3aa3-4be8-8f5a-6e63b2219d00
# ╠═52501b42-4601-4df5-be66-d7ebd56e7404
# ╠═122712f6-61f2-4288-8194-e56d33298059
# ╠═a928ccfd-74d7-4b4b-9617-707b736bb0e1
# ╠═61020487-999d-4af7-b3f5-8fb6ef5258b7
# ╠═e7e4d71e-45f0-4d6c-a973-5e46916ce52c
# ╠═99a423ca-5966-48c0-8828-93368e983ed0
# ╠═d5378cc9-382a-45ba-aabb-0d6c5269c71a
# ╠═84ad3671-4d5c-48e0-a459-1cf6414051e6
# ╠═7da60210-a0b9-4556-95e7-86e429c0a3de
# ╠═7150257c-5bb1-4695-8dd6-829fc7557732
# ╠═a1672a38-9a19-4dec-9b5b-38f053c24382
# ╠═26317fb6-4c61-4ae2-83c8-de8383d463ce
# ╠═de7ba506-26c8-4c55-8806-66c168ecd0a9
# ╠═1dd0676f-0e1b-415a-8209-9b6b34e304b6
# ╠═c115fdc0-3940-470a-ba3e-b0d8d70583b6
# ╠═e8314029-ac3a-4a73-b29d-e033f47f2886
# ╠═e392b03c-835d-425c-b777-d3d31184e089
# ╠═123fe936-9fb2-4084-94b5-29323e2fbace
# ╠═90e9b13d-5172-4b8d-91c1-0f2848d1b1e0
# ╠═f4c87141-8029-4b38-b761-c4a8b9c1c0ba
# ╠═917de72d-7592-41e9-a3ec-304161d6c629
# ╠═523f9965-8533-4a76-9a38-97e8ba55e084
# ╠═4dcfed74-9fb3-4fbf-991f-777026a8bda2
# ╠═d602790f-f9bb-4388-b1e5-094127ca4eb0
# ╠═b6c46a6c-4c57-4bc7-8064-86051f152f60
# ╠═4ba1dc45-984c-465f-a592-890a68dd99f2
# ╠═27e77ef4-7596-4d91-b5b5-86c1044a2b67
# ╠═43a4c5e5-140c-45ea-9b27-859f953aebb6
# ╠═9f4e342c-8973-4faa-81d8-16613cba858d
# ╠═e3cd63e7-5b8d-4b37-b741-cb03806800e5
# ╠═36f74429-156f-4d68-85af-dfa8f90d9f1c
# ╠═c19bd304-4ee0-4eb5-82a7-966370de4b9b
# ╠═453cd265-5617-4db4-b8fc-5a53f2d4fb59
# ╠═ecea9239-4ead-4063-8c12-1ee2511c58cf
# ╠═0837cb26-3a68-47b0-b5b8-70daae283d04
# ╠═f22ed1bd-1ea9-4280-8bbf-dbf1c6f3d90b
# ╠═871f578e-0548-4ff0-a179-53de82b44412
# ╠═7517cd1d-4733-4829-aaa6-3c8a2e20b8e8
# ╠═23b26789-3d64-4165-8198-5513362495e7
# ╠═b1c343e5-753b-415f-8aaf-217cab06b1de
# ╠═e54c6768-fcfc-411b-97a4-927bb8e00a2c
# ╠═18f971be-5a24-47ed-bced-b7370ec2da58
# ╠═9020b6ac-f1b6-40d3-b105-7dd6a62847e0
# ╠═4adb7af1-0cd1-452a-a1d1-c434bb494e2e
# ╠═1c760b9e-2b60-4ee0-a93d-d1eecb48f6bb
# ╠═5413c0f0-6648-4676-a105-0527cbf66ad2
# ╠═fab69841-6fb5-4cc2-bd57-88da4aa1baf9
# ╠═1ce56563-5e96-4116-b69e-7373a960e98f
# ╠═e53652fe-095e-42a8-8fdc-e31645b7bf1e
# ╠═7a2da99c-f713-4fda-b33f-40433cef59ee
# ╠═965e6a07-1b76-4726-ac75-8c334d684f6a
# ╠═05e4d89a-b751-4582-b428-312adbd0af5e
# ╠═7ea44d8c-15f3-42c3-acea-36f0f2c1bc71
# ╠═af4d8058-5fd3-4f91-ad9b-a8db7ffc3b44
# ╠═f899052b-fffe-49df-9e74-11994b30083f
# ╠═cd9af91e-bc8f-4559-8d5d-6938c2606ae8
# ╠═6b92f6f6-b575-4da9-9269-dac9e7260e4e
# ╠═79a51857-5e3f-4f90-845a-50b1f35abc8e
# ╠═982dbd22-28f6-4b2d-84da-36b4305e3f8e
# ╠═3e06f4da-0c9d-42b8-ba02-c9726aaa581c
