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

# ╔═╡ 802599f6-bb71-4ff8-bb60-31646f604986
using Pkg, Revise

# ╔═╡ 4b07d29a-e830-11ee-13b0-9b4b529f3060
using DrWatson

# ╔═╡ d2f770e4-4e9c-4d7a-9111-13b517613881
@quickactivate

# ╔═╡ 263ff2ee-6e7c-40d8-b34b-4cf21b68acac
using SPDE

# ╔═╡ e6823527-5d3e-434b-a25d-a1f5858d6ecf
using DifferentialEquations, LinearAlgebra, SparseArrays, PlutoUI

# ╔═╡ 336f0a29-bf7f-42dd-9333-317056a14c1e
using Symbolics

# ╔═╡ b523f36e-09be-468b-a9ef-8b5199e24cbb
using Plots

# ╔═╡ f81ec869-a1dd-43d5-bdf8-5a4b791188dc
using Random

# ╔═╡ d9b55341-5eb8-496d-836f-48bb823a98c8
using Distributions

# ╔═╡ ff426bf9-9d0f-4921-8937-d6573082b2e8
using Trapz

# ╔═╡ 0387c0e6-d3da-4dba-b8eb-0f374a1c811c
const N = 64

# ╔═╡ dc3fca4c-d712-4940-8157-06844ce17202
begin
	const dx = 1/(N-1)
	const σ(x) = 1.
end

# ╔═╡ f3cd275c-f08e-4a1b-a14c-7fe7c7c08943
p=(dx,σ)

# ╔═╡ 7be62d93-0a53-499b-8aba-9ff836083515


# ╔═╡ 50dcf226-a184-463f-8d9d-02a33eed3ee4


# ╔═╡ c94098dd-32d6-47bf-a532-80d2226d3825
sqrt(1/dx)

# ╔═╡ a33670f4-619c-400a-935d-f272b5fe3840
begin
	u0 = rand(N,N)
	du0 = copy(u0)
end

# ╔═╡ fe97e53f-d454-4c37-a34f-edcea9348594
jac_sparsity = Symbolics.jacobian_sparsity((du,u) -> drift!(du,u,p,0.0),du0,u0)

# ╔═╡ 5e2f9b1f-83a5-4c2d-9697-309b076e44d9
SDE_sparse = SDEFunction(drift!,noise!;jac_prototype=float.(jac_sparsity))

# ╔═╡ a70a78ed-f497-4923-910f-cd309443958a
begin
	u_begin = ones(N,N)
	u_begin[1,:] .= 0; u_begin[end,:] .= 0; u_begin[:,1] .= 0; u_begin[:,end] .=0
end

# ╔═╡ 54b50d8b-7e42-45e3-9a58-234600346634
#nt

# ╔═╡ 0dcc6ee4-9c26-447d-8b2c-ed54f224b6ab
begin
	algo = ImplicitEM(linsolve = KLUFactorization())
	L=1; 
	tmax = 0.3;
	h=2^9
	nx = N;
	nt= 2^16
	dt = tmax/(nt-1); 
	t_idxs = 2^14*dt:dt:tmax
	time_range = (0.0,tmax)
end

# ╔═╡ 06fb9891-6342-4ab3-a72a-b1666e1a52db
length(t_idxs)

# ╔═╡ 6987feef-d4a5-4053-b892-252421777abe
prob = SDEProblem(SDE_sparse,u_begin,time_range,p)

# ╔═╡ a8e6cdcb-1543-48d9-a57b-9301830bad0f
solution=solve(prob,dtmax=dt,saveat=t_idxs,progress=true,algo,maxiters=1e7)

# ╔═╡ c717a670-799d-4936-8dde-24e2130e6a76
t_idxs

# ╔═╡ e9d198a0-c5e5-4c90-adf0-8664836449ca
@bind t Slider(1:1:10000)

# ╔═╡ 3896f0ad-30fa-48e8-8d8a-6a8ec9ef4296
z=solution[:,:,t] 

# ╔═╡ 60b74324-7a6d-4474-bf87-e6dc484f5cd2
begin
	xvec = 0:dx:1
	yvec = 0:dx:1
end

# ╔═╡ cb74f3ca-8c11-4f97-8121-f2e21e79702b
plot(xvec,yvec,z,st=:surface)

# ╔═╡ 95052c7c-be5b-4e66-aa39-4f2216c269ed
plot(xvec,z[:,30])

# ╔═╡ e7e9c72e-4f72-48e6-84b0-508488d904cf
SPDE.partial_integration_2d(solution,dt,dx,L,tmax,1,8,4*dx)

# ╔═╡ 3ba75040-eaf0-42c3-90b9-bf625bb47b5f
solution

# ╔═╡ 35fe46ce-f104-44f3-a020-e3cd8a698b72


# ╔═╡ c21e635e-77bf-4328-bbe8-5d3c6145c763


# ╔═╡ 9ac71496-b4c0-4b6e-a3e1-373c56ffa047
sample

# ╔═╡ 4c2714aa-05d9-4bb3-af47-3c38769e81d7
begin
	results = Channel{Tuple}(Inf)
	Threads.@threads for t in 1:factor:t_len-t_eps-10
		rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		rand_y = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		for i in 1:num_x_samples
			x=rand_x[i]
			y=rand_x[i]
			integrated_view = view(Lu, x:x+x_eps+1,y:y+y_eps+1, t:t+t_eps+1)
			l1,l2,l3 = size(integrated_view)
			integrated=trapz((1:l1,1:l2,1:l3),integrated_view)^2
			u=new_sol[x,y,t]
			put!(results, (u, integrated))
		end
	end
	close(results)
	collected_results = collect(results)
end

# ╔═╡ dc1619ac-c006-40f6-8092-f12ed2ee6a77
begin
	for t in 1:factor:t_len-t_eps-10
	rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
	rand_y = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)
		for i in 1:num_x_samples
			x=rand_x[i]
			y=rand_x[i]
			integrated_view = view(Lu, x:x+x_eps+1,y:y+y_eps01, t:t+t_eps+1)
			l1,l2,l3 = size(integrated_view)
			integrated=trapz((1:l1,1:l2,1:l3),integrated_view)^2
			u=new_sol[x,y,t]
			put!(results, (u, integrated))
		end
	end
end

# ╔═╡ fb0f34c5-a4f7-4e08-bb33-cb6c64998cd8
rand_x = sample(x_eps+2:x_len-x_eps-2,num_x_samples,replace=false)

# ╔═╡ 89c34e78-7632-4fcd-8b5c-65acfad9ac2a
inte_view = view(Lu,32:32+x_eps,32:32+x_eps,1000:1000+t_eps)

# ╔═╡ 15a8bc32-6368-41ca-b7ad-01c44df1e700
l1,l2,l3 = size(inte_view)

# ╔═╡ 2c3cc5e6-a47d-40c4-be85-8d500c0b1289
trapz((1:l1,1:l2,1:l3),inte_view)

# ╔═╡ a979c232-8654-4ccb-9530-902b445e27bf
trapz((l1,l2,l3),inte_view)

# ╔═╡ f14b26e0-2073-4974-a0ac-0919a6e3c42b


# ╔═╡ Cell order:
# ╠═4b07d29a-e830-11ee-13b0-9b4b529f3060
# ╠═d2f770e4-4e9c-4d7a-9111-13b517613881
# ╠═0387c0e6-d3da-4dba-b8eb-0f374a1c811c
# ╠═dc3fca4c-d712-4940-8157-06844ce17202
# ╠═f3cd275c-f08e-4a1b-a14c-7fe7c7c08943
# ╠═263ff2ee-6e7c-40d8-b34b-4cf21b68acac
# ╠═802599f6-bb71-4ff8-bb60-31646f604986
# ╠═e6823527-5d3e-434b-a25d-a1f5858d6ecf
# ╠═7be62d93-0a53-499b-8aba-9ff836083515
# ╠═50dcf226-a184-463f-8d9d-02a33eed3ee4
# ╠═c94098dd-32d6-47bf-a532-80d2226d3825
# ╠═336f0a29-bf7f-42dd-9333-317056a14c1e
# ╠═a33670f4-619c-400a-935d-f272b5fe3840
# ╠═fe97e53f-d454-4c37-a34f-edcea9348594
# ╠═5e2f9b1f-83a5-4c2d-9697-309b076e44d9
# ╠═a70a78ed-f497-4923-910f-cd309443958a
# ╠═06fb9891-6342-4ab3-a72a-b1666e1a52db
# ╠═54b50d8b-7e42-45e3-9a58-234600346634
# ╠═0dcc6ee4-9c26-447d-8b2c-ed54f224b6ab
# ╠═6987feef-d4a5-4053-b892-252421777abe
# ╠═a8e6cdcb-1543-48d9-a57b-9301830bad0f
# ╠═c717a670-799d-4936-8dde-24e2130e6a76
# ╠═3896f0ad-30fa-48e8-8d8a-6a8ec9ef4296
# ╠═e9d198a0-c5e5-4c90-adf0-8664836449ca
# ╠═b523f36e-09be-468b-a9ef-8b5199e24cbb
# ╠═60b74324-7a6d-4474-bf87-e6dc484f5cd2
# ╠═cb74f3ca-8c11-4f97-8121-f2e21e79702b
# ╠═95052c7c-be5b-4e66-aa39-4f2216c269ed
# ╠═e7e9c72e-4f72-48e6-84b0-508488d904cf
# ╠═3ba75040-eaf0-42c3-90b9-bf625bb47b5f
# ╠═35fe46ce-f104-44f3-a020-e3cd8a698b72
# ╠═c21e635e-77bf-4328-bbe8-5d3c6145c763
# ╠═f81ec869-a1dd-43d5-bdf8-5a4b791188dc
# ╠═d9b55341-5eb8-496d-836f-48bb823a98c8
# ╠═9ac71496-b4c0-4b6e-a3e1-373c56ffa047
# ╠═4c2714aa-05d9-4bb3-af47-3c38769e81d7
# ╠═dc1619ac-c006-40f6-8092-f12ed2ee6a77
# ╠═fb0f34c5-a4f7-4e08-bb33-cb6c64998cd8
# ╠═ff426bf9-9d0f-4921-8937-d6573082b2e8
# ╠═89c34e78-7632-4fcd-8b5c-65acfad9ac2a
# ╠═15a8bc32-6368-41ca-b7ad-01c44df1e700
# ╠═2c3cc5e6-a47d-40c4-be85-8d500c0b1289
# ╠═a979c232-8654-4ccb-9530-902b445e27bf
# ╠═f14b26e0-2073-4974-a0ac-0919a6e3c42b
