using DrWatson
push!(LOAD_PATH,srcdir())
@quickactivate
using Revise

using highdim


using SPDE
using DifferentialEquations, LinearAlgebra, SparseArrays

@time sol=highdim.generate_solution(sin)

using Plots

xs,ys,ts = size(sol)
xvec = range(0,1,length=xs)
yvec = range(0,1,length=ys)
#gr()
plot(xvec,yvec,sol[:,:,2004],st=:surface)

df = highdim.test()




