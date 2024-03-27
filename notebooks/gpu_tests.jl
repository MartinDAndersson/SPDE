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
using Revise, gpu

# ╔═╡ 28d43582-65b4-4306-bf10-39f056df1fd0
using highdim

# ╔═╡ bf753f24-7c4a-4813-802e-fd3835f360d5
using PlutoUI

# ╔═╡ 55ea5ddb-20a4-44cb-8177-6f81048c7fae
using Plots

# ╔═╡ a3f88e56-b6b3-4952-a382-9798dd345ae5
using LinearAlgebra

# ╔═╡ 7e17f538-94ad-4714-8907-8cfeadea9bb2
nx = 2^8;nt=2^16

# ╔═╡ 1b99b4b4-9184-48cc-80cd-807a1fac6ae4
sol = test(nx,nt)

# ╔═╡ 255877c2-78f0-4a23-a27a-78a6105f99cb
begin
	xs,ys,ts = size(sol)
	xvec = range(0,1,length=xs)
	yvec = range(0,1,length=ys)
end

# ╔═╡ b070efe4-af9b-46a8-9701-727c101acd25
@bind tp Slider(1:ts)

# ╔═╡ 951a289c-9616-4dc8-ae12-243a5e281cfb
sol_cpu= sol[:,:,tp] |> Array

# ╔═╡ df3a896a-b8d3-409a-af55-99e442400409
sol_cpu[:,1]

# ╔═╡ 87070470-de6f-45bd-809c-145997ad8599
plot(xvec,yvec,sol_cpu,st=:surface,xlims=(0,1),ylims=(0,1),zlims=(0,1),aspect_ratio=:equal)

# ╔═╡ 2f664237-76ba-4614-adb6-d27eaa3d1fd1
plot(xvec,sol_cpu[:,nx ÷ 2])

# ╔═╡ dc14b807-4d16-4ff9-9622-e6ecaf2264b8
sol2=highdim.generate_solution(sin,2^8,2^15)

# ╔═╡ 4bf57111-acb2-46ed-9d81-9e84bb6d5f37
sol_cpu

# ╔═╡ a35c0321-6be5-4c8c-a5be-5a0c13082888
sol2[:,2] |> plot

# ╔═╡ 879af4bf-791a-4cd9-93ae-f211816b7083
#sol_cpu |> plot

# ╔═╡ 4ac9a35e-6d16-4241-841c-a37b28b2dc02
N=2^6

# ╔═╡ ca077c3d-b8d0-4f5c-890f-e3219e11e3fb
begin
	Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                            [1.0 for i in 1:(N - 1)]))
	    Mx[2,1] = 0.
	    Mx[end-1,end] = 0.
		Mx
end

# ╔═╡ 7b5833eb-b22b-4cf7-857d-8e5890d97300
begin
	    My = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                            [1.0 for i in 1:(N - 1)]))
	    My[1,2] = 0.
	    My[end,end-1] = 0.
		My
end

# ╔═╡ 8dae1517-70fe-4aae-bc75-169bedb3dc9c
dt = 1/(nt-1); dx=1/(nx-1)

# ╔═╡ 73c996d9-055b-4c94-b75c-09f518f08037


# ╔═╡ 3693955e-5a1d-4a19-9f3c-18909c77ee47
Array(sol)

# ╔═╡ 863638bf-6dda-456d-8693-84122020dbf7

df=highdim.partial_integration(Array(sol),dt,dx,4,4,4*dx)

# ╔═╡ a990548b-3bb5-4808-83a3-e61ac8888cf0
scatter(df.x,df.y)

# ╔═╡ cd9df95d-419b-467b-8819-77b9c96bcf4b
df.x

# ╔═╡ Cell order:
# ╠═895e87da-eb57-11ee-08e7-07d1e1c510e8
# ╠═28d43582-65b4-4306-bf10-39f056df1fd0
# ╠═a4f6c327-eceb-4f40-8087-9b1ca4ed8e8a
# ╠═bf753f24-7c4a-4813-802e-fd3835f360d5
# ╠═9c2008b0-9bde-4bb3-b4b1-b2a8728b818c
# ╠═31731e32-25eb-4bcf-a4ac-9485ddbb563b
# ╠═cc063a3f-25c9-49fc-95d0-fe32c414759f
# ╠═55ea5ddb-20a4-44cb-8177-6f81048c7fae
# ╠═7e17f538-94ad-4714-8907-8cfeadea9bb2
# ╠═1b99b4b4-9184-48cc-80cd-807a1fac6ae4
# ╠═255877c2-78f0-4a23-a27a-78a6105f99cb
# ╠═951a289c-9616-4dc8-ae12-243a5e281cfb
# ╠═df3a896a-b8d3-409a-af55-99e442400409
# ╠═87070470-de6f-45bd-809c-145997ad8599
# ╠═b070efe4-af9b-46a8-9701-727c101acd25
# ╠═2f664237-76ba-4614-adb6-d27eaa3d1fd1
# ╠═dc14b807-4d16-4ff9-9622-e6ecaf2264b8
# ╠═4bf57111-acb2-46ed-9d81-9e84bb6d5f37
# ╠═a35c0321-6be5-4c8c-a5be-5a0c13082888
# ╠═879af4bf-791a-4cd9-93ae-f211816b7083
# ╠═4ac9a35e-6d16-4241-841c-a37b28b2dc02
# ╠═a3f88e56-b6b3-4952-a382-9798dd345ae5
# ╠═ca077c3d-b8d0-4f5c-890f-e3219e11e3fb
# ╠═7b5833eb-b22b-4cf7-857d-8e5890d97300
# ╠═8dae1517-70fe-4aae-bc75-169bedb3dc9c
# ╠═73c996d9-055b-4c94-b75c-09f518f08037
# ╠═3693955e-5a1d-4a19-9f3c-18909c77ee47
# ╠═863638bf-6dda-456d-8693-84122020dbf7
# ╠═a990548b-3bb5-4808-83a3-e61ac8888cf0
# ╠═cd9df95d-419b-467b-8819-77b9c96bcf4b
