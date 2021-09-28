using ProgressMeter
using .Threads
using Lasso
using NPZ
using Statistics
using PyCall
using JLD
using SparseArrays

py"""
import scipy.sparse as sp

def spload(x):
    return sp.load_npz(x)

def spsave(x, i):
    return sp.save_npz(f"Results/nodal_results_{i}.npz", x)
"""

PyObject(S::SparseMatrixCSC) =
    pyimport("scipy.sparse").csc_matrix((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))

# G = py"spload"("Data/Gcsc.npz")
# G = SparseMatrixCSC(G.shape[1], G.shape[2], G.indptr .+ 1, G.indices .+ 1, G.data)

G = npzread("Data/nodal_G.npy")

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

d = npzread("Data/nodal_wvtdata.npy")
p = Progress(size(d,1) * size(d,2), dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

results = Array{Any,2}(undef, (size(d,1), size(d,2)))

for j = 1:2
    @threads for i = 1:size(d,2)
        results[j,i] = fit_model(G,d[j,i,:],i)
        next!(p)
    end
end

sres1 = reduce(hcat, [x.m.*x.dn for x in results[1,:]])
py"spsave"(PyObject(sres1),1)

sres2 = reduce(hcat, [x.m.*x.dn for x in results[2,:]])
py"spsave"(PyObject(sres2),2)

