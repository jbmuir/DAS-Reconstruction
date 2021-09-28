using ProgressMeter
using .Threads
using Lasso
using NPZ
using Statistics
using PyCall
using SparseArrays

println(ARGS[1])
println("Threads = $(nthreads())")

py"""
import scipy.sparse as sp

def spsave(x, name):
    return sp.save_npz(f"Combined_Results/combined_results_{name}.npz", x)
"""

PyObject(S::SparseMatrixCSC) =
    pyimport("scipy.sparse").csc_matrix((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))

struct InvResult{T,S}
    i::Int
    λmin::T
    dn::T
    m::S
end

function fit_model(G,dv,i)
    dn = std(dv)
    dv ./= dn
    path = fit(LassoPath, G, dv, λminratio = 1e-3, standardize=false, intercept=false, maxncoef=size(G,2)) 
    λmin = path.λ[argmin(aicc(path))]
    m = coef(path; select=MinAICc())
    return InvResult(i, λmin, dn, m)
end

G = npzread("tmp/G.npy")
d = npzread("tmp/data.npy")
p = Progress(size(d,2), dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

results = Array{Any,1}(undef, size(d,2))

@threads for i = 1:size(d,2)
    results[i] = fit_model(G,d[:,i],i)
    next!(p)
end

sres = reduce(hcat, [x.m.*x.dn for x in results])
py"spsave"(PyObject(sres),ARGS[1])
