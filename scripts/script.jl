import Pkg
using Revise
Pkg.activate(".")
using Volatility
using LinearAlgebra, Random, DifferentialEquations, LsqFit
using SparseArrays
using LaTeXStrings, LoopVectorization, DataFrames,CSV, FileIO
using StatsBase
import StatsBase,ProgressLogging
L=1; 
tmax = 0.3;
h=2^9
nx = h;
nt= 2^18
dx = L/(nx-1); 
dt = tmax/(nt-1); 
dt_int=dt
r=dt/dx^2;
σ = x-> sin(x)
#figname = "0.5x"
ϵ = 30dx
samples = 1;
algo = ImplicitEM(linsolve = KLUFactorization())#SROCKEM(strong_order_1=false)#ImplicitEM(linsolve = KLUFactorization())#ImplicitRKMil(linsolve = KLUFactorization()) #SImplicitMidpoint
t0=tmax;
u_begin = 6*ones(nx); u_begin[1] = 0; u_begin[end] = 0;
#u_begin = Float32.(u_begin)
#save_dt = dx/10
#x_eps = (ϵ ÷ dx) |> Int 
#t_eps = (ϵ ÷ dt) |> Int
#mid = nx ÷ 2 |> Int 


t_idxs = 2^14*dt:dt:12


df = experiment_3(u_begin,nx,20,dt,dx,L,tmax,algo,t_idxs,[(x->sin(x),"sin(x)"),(x->x,"x"),(x->sin(pi*x),"sin(pix)"),
    (x->0.1 + 0.3*x + sin(x),"0.1 + 0.3*x + sin(x)")])
CSV.write("data.csv", df)

