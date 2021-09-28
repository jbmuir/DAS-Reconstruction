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

Gzigzag = npzread("Kernels/G_zigzag.npy")
Gspiral = npzread("Kernels/G_spiral.npy")
Gcross = npzread("Kernels/G_cross.npy")
Grandom = npzread("Kernels/G_random.npy")

dzigzag = npzread("Testing/zigzag.npy")
dspiral = npzread("Testing/spiral.npy")
dcrossing = npzread("Testing/crossing.npy")
drandom = npzread("Testing/random.npy")

p = Progress(4*size(dzigzag,2), dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

for (G, d, name) in zip([Gzigzag, Gspiral, Gcross, Grandom], [dzigzag, dspiral, dcrossing, drandom], ["zigzag", "spiral", "crossing", "random"])
        results = Array{Any,1}(undef, size(d,2))
        uresults = Array{Any,1}(undef, size(d,2))
    
        G = G / std(G)

        @threads for j = 1:size(d,2)
            results[j] = fit_model(G, d[:,j], j)
            next!(p)
        end

        sres = reduce(hcat, [x.m.*x.dn for x in results])

        npzwrite("Testing/$(name)_res.npy", Array{Float64}(sres))
 
end



