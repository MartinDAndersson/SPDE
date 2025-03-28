"""
    gpu

Module implementing GPU-accelerated SPDE solvers using CUDA.

This module provides functions for solving stochastic heat equations on
GPUs, enabling much faster computation for 2D problems and large systems.
It leverages CUDA.jl for parallel computation on NVIDIA GPUs.
"""
module gpu
using DifferentialEquations, LinearAlgebra
using CUDA
export test
"""
    test(N, nt)

Run a GPU-accelerated 2D stochastic heat equation solver.

# Arguments
- `N::Int`: Number of spatial grid points in each dimension (NxN grid)
- `nt::Int`: Number of time steps

# Returns
- `Array{Float32}`: Solution array transferred from GPU to CPU

This function sets up and solves a 2D stochastic heat equation with a 
state-dependent diffusion coefficient σ(u) = sqrt(dx)*sin(πu) using CUDA.
It implements a finite difference scheme with fixed boundary conditions
(u=0 at boundaries) and uses the SROCK1 algorithm for time integration.

The spatial discretization uses a standard 5-point stencil for the 2D Laplacian.
"""
function test(N,nt)
    #N=2^7
    #nt = 2^17
    dt = 1/(nt-1)

    Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                            [1.0 for i in 1:(N - 1)]))

                            
    My = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                            [1.0 for i in 1:(N - 1)]))
    u0 = 6*ones(N, N)
    u0[end,:] .= 0
    u0[1,:] .= 0
    u0[:,1] .= 0
    u0[:,end] .= 0
    L=1
    dx = L/(N-1)
    #My[end,:] .= 0
    Mx[2,1] = 0.
    Mx[end-1,end] = 0.
    My[1,2] = 0.
    My[end,end-1] = 0.
    MyA = zeros(N, N);
    AMx = zeros(N, N);
    #AMy = zeros(N, N);
    #DA = zeros(N, N);
    D = 0.5*1/dx^2
    Dnoise = sqrt(1/dx^2)

    function f(du,u,p,t)
        mul!(MyA,My,u)
        mul!(AMx,u,Mx)
        @. du = D*(MyA)
        @. du[1,:] = 0
        @. du[end,:] = 0
        @. du[:,1] = 0
        @. du[:,end] = 0
    end
    #prob = ODEProblem(f, u0, (0.0, 100.0))
   # @time sol = solve(prob, ROCK2(), progress = true, save_everystep = false, save_start = false)
    σ(x) = sqrt(dx)*sin(pi*x)
    gMx = CuArray(Float32.(Mx))
    gMy = CuArray(Float32.(Mx))
    # gα₁ = CuArray(Float32.(α₁))
    gu0 = CuArray(Float32.(u0))
    gMyA = CuArray(zeros(Float32, N, N))
    AgMx = CuArray(zeros(Float32, N, N))
    edges = ones(N,N)
    edges[1,:] .= 0
    edges[end,:] .= 0
    edges[:,1] .= 0
    edges[:,end] .= 0
    gedges = CuArray(Float32.(edges))
    riez = [sqrt(abs(x-y)) for x in 0:dx:1,y in 0:dx:1]*2
    riez = CuArray(Float32.(riez))
    function gf(du,u,p,t)
        mul!(gMyA,gMy,u)
        mul!(AgMx,u,gMx)
        @. du = D*(gMyA+AgMx)
        @. du = gedges * du 
    end

    function gnoise(du,u,p,t)
        @. du = σ(u) * Dnoise * riez
    end

    prob2 = ODEProblem(gf, gu0, (0.0, 0.3))
    CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
    #@time sol = solve(prob2, ROCK2(), progress = true,save_everystep = false, dt = dt,save_start = false);
    prob = SDEProblem(gf, gnoise, gu0, (0.0, 0.3))
    #tquot = 
    t_idxs = 0.05:dt:0.3
    #save_dt = 
    eps=30
    xmid = N ÷ 2
    ymid = (N - (N ÷ 8))
    x_idxs= CartesianIndices((xmid-eps:xmid+eps,ymid-eps:ymid+eps))
    @time sol = solve(prob,saveat=t_idxs,dt=dt,save_idxs=x_idxs, SROCK1(),progress=true);

    return Array(sol)
end

"""
    test2(N, nt)

Run a GPU-accelerated 1D stochastic heat equation solver.

# Arguments
- `N::Int`: Number of spatial grid points 
- `nt::Int`: Number of time steps

# Returns
- `SciMLBase.RODESolution`: Solution object from the SDE solver

This function is a 1D variant of the `test` function, solving a one-dimensional
stochastic heat equation with a state-dependent diffusion coefficient σ(u) = sin(πu).
It uses CUDA for GPU acceleration and implements a finite difference scheme with
fixed boundary conditions (u=0 at boundaries).

This implementation is useful for comparison with the 2D version to study
performance scaling and for simpler test cases.
"""
function test2(N,nt)
    #N=2^7
    #nt = 2^17
    dt = 1/(nt-1)

    Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                            [1.0 for i in 1:(N - 1)]))

                            

    u0 = 6*ones(N)
    u0[1] = 0
    u0[end] = 0
    L=1
    dx = L/(N-1)
    #My[end,:] .= 0
    Mx[1,2] = 0.
    Mx[end,end-1] = 0.
    MxA = zeros(N, N);
    #AMy = zeros(N, N);
    #DA = zeros(N, N);
    D = 0.5*1/dx^2
    Dnoise = sqrt(1/dx)
    #prob = ODEProblem(f, u0, (0.0, 100.0))
   # @time sol = solve(prob, ROCK2(), progress = true, save_everystep = false, save_start = false)
    σ(x) = sin(pi*x)
    gMx = CuArray(Float32.(Mx))
    # gα₁ = CuArray(Float32.(α₁))
    gu0 = CuArray(Float32.(u0))
    gMxA = CuArray(zeros(Float32, N))
    edges = ones(N)
    edges[1] = 0
    edges[end] = 0
    gedges = CuArray(Float32.(edges))
    function gf(du,u,p,t)
        mul!(gMxA,gMx,u)
        @. du = D*(gMxA)
        @. du = gedges * du 
    end

    function gnoise(du,u,p,t)
        @. du = σ(u) * Dnoise 
    end

    CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
    #@time sol = solve(prob2, ROCK2(), progress = true,save_everystep = false, dt = dt,save_start = false);
    prob = SDEProblem(gf, gnoise, gu0, (0.0, 0.3))
    #tquot = 
    t_idxs = 0.05:dt:0.3
    #save_dt = 
    eps=30
    xmid = N ÷ 2
    ymid = (N - (N ÷ 8))
    #x_idxs= xmid - eps: xmid + eps
    @time sol = solve(prob,saveat=t_idxs,dt=dt, SROCK1(),progress=true) 

    return sol
end

end