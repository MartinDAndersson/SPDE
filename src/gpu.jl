module gpu
using DifferentialEquations, LinearAlgebra
using CUDA
export test
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
    Dnoise = sqrt(1/dx)

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
    σ(x) = x
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
    function gf(du,u,p,t)
        mul!(gMyA,gMy,u)
        mul!(AgMx,u,gMx)
        @. du = D*(gMyA+AgMx)
        du .= gedges .* du 
    end

    function gnoise(du,u,p,t)
        @. du = σ.(u) * Dnoise
    end

    prob2 = ODEProblem(gf, gu0, (0.0, 0.3))
    CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
    #@time sol = solve(prob2, ROCK2(), progress = true,save_everystep = false, dt = dt,save_start = false);
    prob = SDEProblem(gf, gnoise, gu0, (0.0, 0.3))
    #tquot = 
    t_idxs = 0.1:dt:0.3
    #save_dt = 
    @time sol = solve(prob,saveat=t_idxs,dt=dt, SROCK1(),progress=true);

    return Array(sol)
end

end