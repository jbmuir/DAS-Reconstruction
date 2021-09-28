using ProgressMeter
using .Threads
using Lasso
using NPZ
using Statistics
using Random
using StatsBase
using LinearAlgebra

struct InvResult{T,S}
    i::Int
    λmin::T
    dn::T
    m::S
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

p = Progress(n_runs*(length(s_size)-1)+1, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

for B in s_size
    pvec = npzread("IncoherenceResults/res_inc_$B.npy")

    for i in 1:(B!=64 ? n_runs : 1)
        rng = MersenneTwister(i+50)
        sensors = sample(rng, 1:64, aweights(pvec), B, replace=false)
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



