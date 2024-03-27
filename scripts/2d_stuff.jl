using DrWatson
push!(LOAD_PATH,srcdir())
@quickactivate
using Revise
using OhMyREPL
using highdim
using Plots
using AlgebraicMultigrid
function algebraicmultigrid(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
    if newW === nothing || newW
        Pl = aspreconditioner(ruge_stuben(convert(AbstractMatrix, W)))
    else
        Pl = Plprev
    end
    Pl, nothing
end
algo=ImplicitEM(linsolve = KLUFactorization())
#using SPDE
using DifferentialEquations, LinearAlgebra, SparseArrays


dt=1/2^15
t_idxs = 2^11*dt:dt:0.4

@time E=highdim.generate_solution(sin)
@time sol=solve(E,dtmax=dt,saveat=t_idxs,progress=true,algo,maxiters=1e7)



xs,ys,ts = size(sol)
xvec = range(0,1,length=xs)
yvec = range(0,1,length=ys)
plotlyjs()
plot(xvec,yvec,sol[:,:,end],st=:surface)

#mach,df = highdim.test()

#fun = highdim.mach_to_func(mach)


#plot(fun,xlims=(0,4))

histogram(df.x)