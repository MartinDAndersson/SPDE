using DrWatson
@quickactivate

using DifferentialEquations, LinearAlgebra
using CUDA

N=2^7
nt = 2^17
dt = 1/nt

const Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                             [1.0 for i in 1:(N - 1)]))

                             
const My = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                             [1.0 for i in 1:(N - 1)]))
u0 = ones(N, N)
u0[end,:] .= 0
u0[1,:] .= 0
u0[:,1] .= 0
u0[:,end] .= 0
const L=1.
const dx = L/(N-1)
const MyA = zeros(N, N);
const AMx = zeros(N, N);
const AMy = zeros(N, N);
const DA = zeros(N, N);
const D = 0.5*1/dx^2

function f(du,u,p,t)
    mul!(MyA,My,u)
    mul!(AMx,u,Mx)
    @. du = D*(MyA)
end
prob = ODEProblem(f, u0, (0.0, 100.0))
#@time sol = solve(prob, ROCK2(), progress = true, save_everystep = false, save_start = false)

const gMx = CuArray(Float32.(Mx))
const gMy = CuArray(Float32.(Mx))
#const gα₁ = CuArray(Float32.(α₁))
gu0 = CuArray(Float32.(u0))
const gMyA = CuArray(zeros(Float32, N, N))
const AgMx = CuArray(zeros(Float32, N, N))
function gf(du,u,p,t)
    mul!(gMyA,gMy,u)
    mul!(AgMx,u,gMx)
    @. du = D*(gMyA+AgMx)
end

prob2 = ODEProblem(gf, gu0, (0.0, 10.0))
CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
@time sol = solve(prob2, ROCK2(), progress = true,save_everystep = false, dt = dt,save_start = false);