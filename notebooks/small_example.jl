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

# ╔═╡ 49cbdcdc-eb6e-11ee-2c8c-fb46a6dd2099
using DrWatson

# ╔═╡ 5b93904b-c46b-4285-87a0-f703ef3b0182
@quickactivate

# ╔═╡ 56bd2b02-9b69-4531-9cf5-b6d30c7ecae7
using Revise,SPDE, MLJ,Plots,PlutoUI

# ╔═╡ ebe839c2-c839-40e4-a143-b3ee8dae2918


# ╔═╡ 5b26d15a-2440-41b4-a17b-4b8a4325332d
dic=Dict(
		"epsilon" => 16,
		"xquot" => 8,
		"tquot" => 16,
		"sigma" => "sigma3",
		"sample" => 1
	)

# ╔═╡ 3ddb6aee-35ef-43e8-ae80-641313d626f5
σ(x) = sin(x)

# ╔═╡ af7582f4-e663-4ce5-a990-e29f77c3bf67
sol=SPDE.generate_solution(σ)

# ╔═╡ 958263bf-e9e0-42e7-bfa0-3cf139eae735
xs,ts = size(sol)

# ╔═╡ e3d7293e-e7ad-4f09-99ee-705f32aa23ef
xvec = range(0,1,length=xs)

# ╔═╡ 31cb39c3-0121-4998-9b98-6b4bd3a0c7aa
@bind t Slider(1:ts)

# ╔═╡ afa7f850-cb0d-40b3-ae1a-c441ba9a46cd
plot(xvec,sol[:,t])

# ╔═╡ f811b347-3234-460a-9897-0a01a8f9a3c5
mach,fulld,truth,df=SPDE.main_exp(dic)

# ╔═╡ 4f73fcd2-1f48-4c89-a35e-5269099066df
report(mach)

# ╔═╡ bb42cfc3-833a-4916-b3ac-72c4f48dd246
begin
	    L=1
	    tmax = 0.3;
		#σ = x-> 0.5*sin(2pi*x/6)
	    h=2^9
	    nx = h;
		nt= 2^18
	    dx = L/(nx-1); 
		dt = tmax/(nt-1);
	    tmax = 0.3; 
end

# ╔═╡ dd2a20df-a457-4b1c-af49-b5604dfcd9c1
SPDE.partial_integration(Matrix(sol),dt,dx,L,tmax,2,2,2*dx)

# ╔═╡ 833e4c76-d840-4a6b-ba5b-9b3fd06a617c
fmach = SPDE.mach_to_func(mach)

# ╔═╡ 47b3e71f-47cb-4c13-a366-c63e6a86bc82
begin
	plot(fmach,xlims=(0,6))
	plot!(truth)
end

# ╔═╡ Cell order:
# ╠═49cbdcdc-eb6e-11ee-2c8c-fb46a6dd2099
# ╠═5b93904b-c46b-4285-87a0-f703ef3b0182
# ╠═ebe839c2-c839-40e4-a143-b3ee8dae2918
# ╠═56bd2b02-9b69-4531-9cf5-b6d30c7ecae7
# ╠═5b26d15a-2440-41b4-a17b-4b8a4325332d
# ╠═3ddb6aee-35ef-43e8-ae80-641313d626f5
# ╠═af7582f4-e663-4ce5-a990-e29f77c3bf67
# ╠═958263bf-e9e0-42e7-bfa0-3cf139eae735
# ╠═e3d7293e-e7ad-4f09-99ee-705f32aa23ef
# ╠═31cb39c3-0121-4998-9b98-6b4bd3a0c7aa
# ╠═afa7f850-cb0d-40b3-ae1a-c441ba9a46cd
# ╠═f811b347-3234-460a-9897-0a01a8f9a3c5
# ╠═4f73fcd2-1f48-4c89-a35e-5269099066df
# ╠═bb42cfc3-833a-4916-b3ac-72c4f48dd246
# ╠═dd2a20df-a457-4b1c-af49-b5604dfcd9c1
# ╠═833e4c76-d840-4a6b-ba5b-9b3fd06a617c
# ╠═47b3e71f-47cb-4c13-a366-c63e6a86bc82
