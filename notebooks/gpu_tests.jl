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

# ╔═╡ 9c2008b0-9bde-4bb3-b4b1-b2a8728b818c
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ 895e87da-eb57-11ee-08e7-07d1e1c510e8
using DrWatson

# ╔═╡ a4f6c327-eceb-4f40-8087-9b1ca4ed8e8a
@quickactivate

# ╔═╡ 31731e32-25eb-4bcf-a4ac-9485ddbb563b
push!(LOAD_PATH,srcdir())

# ╔═╡ cc063a3f-25c9-49fc-95d0-fe32c414759f
using Revise, gpu,highdim,Plots, MLJ,PlutoUI

# ╔═╡ 457d7b91-73a8-4758-b50a-a41c9efc9257
using JLD2

# ╔═╡ a3f88e56-b6b3-4952-a382-9798dd345ae5
using LinearAlgebra

# ╔═╡ 0f3565ad-eae9-41d0-bc98-4dfef01c01f2
using DataFrames

# ╔═╡ 7e17f538-94ad-4714-8907-8cfeadea9bb2
nx = 2^9;nt=2^18

# ╔═╡ 73b44776-5a29-4546-8c01-10df19d766ca
sol=gpu.test(nx,nt)

# ╔═╡ a21050d2-89ba-4737-b1fb-32b4afb63cb4
sol[:,:,1][1:2]

# ╔═╡ eddaae96-f739-4cbd-9d80-84d5673adcdc
sol |> size

# ╔═╡ 6b771b26-f9fb-4be8-8661-7dcabf16d265
Revise.retry()

# ╔═╡ 6c1619f0-8d28-4640-b6aa-34e60078d16e
eachindex(rand(4,4))

# ╔═╡ dc14b807-4d16-4ff9-9622-e6ecaf2264b8
#@time sol2=highdim.generate_solution(σ,nx,nt);
#JLD2.@load "sol.jld" sol

# ╔═╡ 3d6db4f4-0a4d-4d16-8912-3591c2aea064
#sol=highdim.gpu_generation(σ,nx,nt);

# ╔═╡ b40867b6-e8d0-4e70-9b5e-767b5ea83227
#JLD2.@load "sol.jld" sol

# ╔═╡ efe8f119-8509-4742-a096-3d990cac70e3
size(sol)

# ╔═╡ 815e2320-6506-4e1e-88ca-7729e01304fb
CartesianIndices((10:12,10:12)) *2

# ╔═╡ 6cbd5ebb-a37a-4c60-83e7-659c248d0c46
#res=highdim.partial_new(sol,dt,dx,1)

# ╔═╡ 0ed3adb2-48c6-45c9-9755-997bf4ccb1ef
#Lu = highdim.L_op((@view sol[:,:,:]),dt,dx)

# ╔═╡ bd56f22a-91a6-4b16-9eb5-f4583bdbf97e
@view sol[:,:,:]

# ╔═╡ 0fe5067a-ce64-4632-bda0-f3f1f8904eed
#similar(@view sol[:,:,:])

# ╔═╡ 7514196d-d739-4499-afa0-e65ea2bab9db
sol

# ╔═╡ 1b99b4b4-9184-48cc-80cd-807a1fac6ae4
#sol = test(nx,nt)

# ╔═╡ 1d8cd281-0f1d-4640-bdde-0200bf25e2a0
size(sol)

# ╔═╡ 96c30810-74ec-408e-bd57-1412c90a30ac
nx

# ╔═╡ 255877c2-78f0-4a23-a27a-78a6105f99cb
begin
	xs,ys,ts = size(sol)
	xvec = range(0,1,length=xs)
	yvec = range(0,1,length=ys)
end

# ╔═╡ b070efe4-af9b-46a8-9701-727c101acd25
@bind tp Slider(1:ts)

# ╔═╡ 87070470-de6f-45bd-809c-145997ad8599
plot(xvec,yvec,sol[:,:,tp],st=:surface,xlims=(0,1),ylims=(0,1),zlims=(0,4),aspect_ratio=:equal)

# ╔═╡ e007f5d0-2579-4d63-adbf-8ce7a0d85a32
sol

# ╔═╡ 4e8a56b7-376d-4fd0-8bf0-e932716286a4
#@save "sol.jld" sol

# ╔═╡ 5bdd75a8-91f1-49b7-8e85-cb3f5e2f13c2
sol |> size

# ╔═╡ cc7fb21d-0aba-491c-a829-0777446d16db
sol[:,:,1]

# ╔═╡ aeb2323a-7367-45cd-87dc-4152f22d5cbf
size(sol)

# ╔═╡ be137b1e-e274-472d-93db-07574205c724
sol |>size

# ╔═╡ c57d5eb3-1102-4665-bb96-deb73521469a


# ╔═╡ 879af4bf-791a-4cd9-93ae-f211816b7083
#sol_cpu |> plot

# ╔═╡ 5879e322-0ab8-4516-a67c-b76bcfc33321
#L_op=highdim.L_op(Array(sol),dt,dx)

# ╔═╡ 4e7c3b78-c012-4a88-8fc5-744ddb85c125
begin
	#scatter(sol[2,2,:],L_op[2,2,:].^2 .* dt*dx^2)
	#plot!(truth)
	#plot!(mach2)
end

# ╔═╡ f7c82d66-a578-4acd-b257-ffcb38598e0d
#df2=DataFrame(:x=>sol[nx ÷ 2,nx÷2,:], :y=>L_op[nx ÷ 2,nx ÷ 2,:].^2*dt*dx^2)

# ╔═╡ 1e4dd963-0b3b-49be-a382-f715f1ec0072
sol[nx ÷ 2, nx ÷ 2,:] |> histogram

# ╔═╡ 8dae1517-70fe-4aae-bc75-169bedb3dc9c
dt = 0.3/(nt-1); dx=1/(nx-1)

# ╔═╡ c4f3b30c-a24e-4a3c-adfb-13797205dbc0
begin
	σ(x) = sqrt(dx)*0.5*x
	truth(x) = σ(x)^2
end

# ╔═╡ 04c7555f-9e01-4c29-ad0e-ea9c74df21a2
riez = [sqrt(abs(x-y)) for x in 0:dx:1,y in 0:dx:1][end - (nx ÷ 8), nx ÷ 8]

# ╔═╡ 8eb096d5-7466-4b94-b5a6-e401521a7797
sqrt(dx)

# ╔═╡ 863638bf-6dda-456d-8693-84122020dbf7
df=highdim.partial_integration((@view sol[3:15,3:15,:]),dt,dx,1,1,1*dx)

# ╔═╡ 89ac129d-9a4d-4d10-b1b6-e22f6bf936ba
scatter(df.x,df.y)

# ╔═╡ 56e6e7dc-cfd2-498c-aaa3-dad87f970615
mach2 = highdim.train_tuned(df) |> highdim.mach_to_func

# ╔═╡ 1d9bd154-d63f-4383-9cf6-e4644edc023a
begin
	L_op_scaled = dt*L_op
	scatter(sol[16,16,:],L_op_scaled[16,16,:].^2)
	plot!(x->truth(x),xlims=(0,3))
end

# ╔═╡ ce11170d-c651-48e0-a768-7c5eb1e6a89e
begin
	xrange = 0:dx:1
	yrange = 0:dx:1
	[x-y for x in xrange, y in yrange]
end

# ╔═╡ 651cc2a7-64b4-4f7a-84af-5376a0ab6c05


# ╔═╡ ba98f24b-1472-44ae-a68a-edd2199a57a6
s

# ╔═╡ 6629c61c-f467-4488-85a6-3c759f5dfa22
mach=highdim.train_tuned(df)

# ╔═╡ f4e56680-d573-4bbc-bdf6-a2bea55fabdb
report(mach)

# ╔═╡ 158b71f8-b21a-4064-a4de-7cbad62e625f
begin
	est_fun = highdim.mach_to_func(mach)
	quotient(x) = est_fun(x) / truth(x)
end

# ╔═╡ 5b2fdff8-d2a5-4ebe-be9e-659e3daea585
quot_mean = quotient.(0.2:0.01:3) |> median

# ╔═╡ 718367ff-1dca-4cef-a904-24404e3a8a3b
begin
	#scatter(df.x,df.y,alpha=0.1,xlims=(0,3))
	#plot(est_fun,xlims=(0,4))
	plot(truth)
	plot!(x->est_fun(x)/quot_mean,xlims=(0,6))
	#plot!(quotient,xlims=(0.5,3))
	#scatter!(df.x,df.y,alpha=0.1,xlims=(0,3))
	#plot!(x->(0.1x)^2)
end

# ╔═╡ 0e19a74a-350d-4253-91c7-c66ae53196ad
#solcpu = highdim.generate_solution(sin,2^4,2^16)

# ╔═╡ cd9df95d-419b-467b-8819-77b9c96bcf4b
begin
	highdim.mach_to_func(mach) |> plot
	#plot!(x->x^2)
end

# ╔═╡ e44d6d2b-8d97-4e6d-a492-9018e33fee0c


# ╔═╡ Cell order:
# ╠═895e87da-eb57-11ee-08e7-07d1e1c510e8
# ╠═a4f6c327-eceb-4f40-8087-9b1ca4ed8e8a
# ╠═9c2008b0-9bde-4bb3-b4b1-b2a8728b818c
# ╠═31731e32-25eb-4bcf-a4ac-9485ddbb563b
# ╠═cc063a3f-25c9-49fc-95d0-fe32c414759f
# ╠═a21050d2-89ba-4737-b1fb-32b4afb63cb4
# ╠═7e17f538-94ad-4714-8907-8cfeadea9bb2
# ╠═eddaae96-f739-4cbd-9d80-84d5673adcdc
# ╠═c4f3b30c-a24e-4a3c-adfb-13797205dbc0
# ╠═73b44776-5a29-4546-8c01-10df19d766ca
# ╠═6b771b26-f9fb-4be8-8661-7dcabf16d265
# ╠═6c1619f0-8d28-4640-b6aa-34e60078d16e
# ╠═dc14b807-4d16-4ff9-9622-e6ecaf2264b8
# ╠═3d6db4f4-0a4d-4d16-8912-3591c2aea064
# ╠═b40867b6-e8d0-4e70-9b5e-767b5ea83227
# ╠═efe8f119-8509-4742-a096-3d990cac70e3
# ╠═815e2320-6506-4e1e-88ca-7729e01304fb
# ╠═6cbd5ebb-a37a-4c60-83e7-659c248d0c46
# ╠═0ed3adb2-48c6-45c9-9755-997bf4ccb1ef
# ╠═bd56f22a-91a6-4b16-9eb5-f4583bdbf97e
# ╠═0fe5067a-ce64-4632-bda0-f3f1f8904eed
# ╠═f4e56680-d573-4bbc-bdf6-a2bea55fabdb
# ╠═7514196d-d739-4499-afa0-e65ea2bab9db
# ╠═1b99b4b4-9184-48cc-80cd-807a1fac6ae4
# ╠═1d8cd281-0f1d-4640-bdde-0200bf25e2a0
# ╠═96c30810-74ec-408e-bd57-1412c90a30ac
# ╠═04c7555f-9e01-4c29-ad0e-ea9c74df21a2
# ╠═255877c2-78f0-4a23-a27a-78a6105f99cb
# ╠═87070470-de6f-45bd-809c-145997ad8599
# ╟─b070efe4-af9b-46a8-9701-727c101acd25
# ╠═e007f5d0-2579-4d63-adbf-8ce7a0d85a32
# ╠═4e8a56b7-376d-4fd0-8bf0-e932716286a4
# ╠═457d7b91-73a8-4758-b50a-a41c9efc9257
# ╠═8eb096d5-7466-4b94-b5a6-e401521a7797
# ╠═5bdd75a8-91f1-49b7-8e85-cb3f5e2f13c2
# ╠═cc7fb21d-0aba-491c-a829-0777446d16db
# ╠═aeb2323a-7367-45cd-87dc-4152f22d5cbf
# ╠═863638bf-6dda-456d-8693-84122020dbf7
# ╠═be137b1e-e274-472d-93db-07574205c724
# ╠═89ac129d-9a4d-4d10-b1b6-e22f6bf936ba
# ╠═c57d5eb3-1102-4665-bb96-deb73521469a
# ╠═879af4bf-791a-4cd9-93ae-f211816b7083
# ╠═a3f88e56-b6b3-4952-a382-9798dd345ae5
# ╠═5879e322-0ab8-4516-a67c-b76bcfc33321
# ╠═4e7c3b78-c012-4a88-8fc5-744ddb85c125
# ╠═0f3565ad-eae9-41d0-bc98-4dfef01c01f2
# ╠═f7c82d66-a578-4acd-b257-ffcb38598e0d
# ╠═56e6e7dc-cfd2-498c-aaa3-dad87f970615
# ╠═1e4dd963-0b3b-49be-a382-f715f1ec0072
# ╠═1d9bd154-d63f-4383-9cf6-e4644edc023a
# ╠═8dae1517-70fe-4aae-bc75-169bedb3dc9c
# ╠═ce11170d-c651-48e0-a768-7c5eb1e6a89e
# ╠═651cc2a7-64b4-4f7a-84af-5376a0ab6c05
# ╠═5b2fdff8-d2a5-4ebe-be9e-659e3daea585
# ╠═718367ff-1dca-4cef-a904-24404e3a8a3b
# ╠═158b71f8-b21a-4064-a4de-7cbad62e625f
# ╠═ba98f24b-1472-44ae-a68a-edd2199a57a6
# ╠═6629c61c-f467-4488-85a6-3c759f5dfa22
# ╠═0e19a74a-350d-4253-91c7-c66ae53196ad
# ╠═cd9df95d-419b-467b-8819-77b9c96bcf4b
# ╠═e44d6d2b-8d97-4e6d-a492-9018e33fee0c
