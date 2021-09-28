using ProgressMeter
using .Threads
using Lasso
using NPZ
using Statistics
using Random
using StatsBase
using LinearAlgebra
import Base: isless

struct TestResult
    i::Int64
    f::Float64
end
isless(a::TestResult, b::TestResult) = isless(a.f, b.f)

struct InvResult{T,S}
    i::Int
    λmin::T
    dn::T
    m::S
end

T = Float64
X1 = npzread("Kernels/G1.npy")
X2 = npzread("Kernels/G2.npy")
X3 = npzread("Kernels/G4.npy")
nfactor = npzread("Testing/nfactor.npy")

ϵ = 1e-2
λ = 1.0
lnorm = -log(ϵ)*(2*size(X1,2))

X = [X3 zeros(T, size(X3));
     zeros(T, size(X3)) X3;
     nfactor*X1 nfactor*X2]
Xn = maximum(sqrt.(sum(X.^2, dims=2)))
X ./= Xn

n = size(X3, 1)
ndas = size(X1, 1)

function Ml(μ, X, ϵ)
    M = ϵ*I + X'*Diagonal(μ)*X
    Mc = cholesky(Hermitian(M))
    l = μ.^2 .*diag(X*(Mc\X'))
    return (M, Mc, l)
end

Ml(μ) = Ml(μ, X, ϵ)

function f(μ, λ, lnorm, n)
    M, Mc, l = Ml(μ) 
    l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
    return -logdet(M) + λ*maximum(l2) - lnorm
end

f(μ) = f(μ, λ, lnorm, n)

function testf(sta, i)
    u_test = vcat(zeros(64+64), ones(size(X1,1)))
    u_test[sta] .= 1.0
    u_test[sta.+64] .= 1.0
    return TestResult(i, f(u_test))
end

function fit_model(G,dv,i)
    dn = std(dv)
    dv ./= dn
    path = fit(LassoPath, G, dv, λminratio = 1e-5, standardize=false, intercept=false, maxncoef=size(G,2)) 
    λmin = path.λ[argmin(aicc(path))]
    m = coef(path; select=MinAICc())
    return InvResult(i, λmin, dn, m)
end

T = Float64

G1 = npzread("Kernels/G1.npy")
G2 = npzread("Kernels/G2.npy")
G3 = npzread("Kernels/G4.npy")

d1 = npzread("Testing/das_wvt_data.npy")
d2 = npzread("Testing/nodal_wvt_data_x.npy")
d3 = npzread("Testing/nodal_wvt_data_y.npy")
nfactor = npzread("Testing/nfactor.npy")

G = [nfactor*G1 nfactor*G2;
     G3 zeros(T, size(G3));
     zeros(T, size(G3)) G3]

Gn = std(G)

G ./= Gn

d = [d1; d2; d3]

s_size = [4, 8, 16, 32, 64]
n_runs = 10
n_selecting = 1000

p = Progress(n_runs*(length(s_size)-1)+1, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

for B in s_size
    pvec = npzread("IncoherenceResults/res_inc_$B.npy")

    for i in 1:(B!=64 ? n_runs : 1)
        rng = MersenneTwister(i+50)
        
        sensors_vec = [sample(rng, 1:64, aweights(pvec), B, replace=false) for k = 1:n_selecting]
        sensors_res = [testf(sta, i) for (i, sta) in enumerate(sensors_vec)]
        best = minimum(sensors_res)
        sensors = sensors_vec[best.i]
        
        usensors = sample(rng, 1:64, B, replace=false)
        μ = [(i in sensors ? 1.0 : 0.0) for i in 1:64]
        uμ = [(i in usensors ? 1.0 : 0.0) for i in 1:64]
        μl = vcat(ones(size(G1, 1)), μ, μ)
        uμl = vcat(ones(size(G1, 1)), uμ, uμ)
        Gμ = Diagonal(μl)*G
        Guμ = Diagonal(uμl)*G
        results = Array{Any,1}(undef, size(d,2))
        uresults = Array{Any,1}(undef, size(d,2))

        @threads for j = 1:size(d,2)
            results[j] = fit_model(Gμ, d[:,j], j)
            uresults[j] = fit_model(Guμ, d[:,j], j)
        end

        sres = reduce(hcat, [x.m.*x.dn for x in results])
        usres = reduce(hcat, [x.m.*x.dn for x in uresults])

        npzwrite("Testing/LassoOutputInc/32B_$(B)_run_$i.npy", Array{Float64}(sres))
        npzwrite("Testing/LassoOutputInc/32B_$(B)_urun_$i.npy", Array{Float64}(usres))
        npzwrite("Testing/LassoOutputInc/32B_$(B)_run_$(i)_sensors.npy", sensors)
        npzwrite("Testing/LassoOutputInc/32B_$(B)_urun_$(i)_sensors.npy", usensors)
        next!(p)
    end
end



