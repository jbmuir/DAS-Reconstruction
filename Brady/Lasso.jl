
using Distributed
addprocs(24)
@everywhere using ProgressMeter

@everywhere using Lasso
using NPZ
@everywhere using Statistics
using PyCall
@everywhere using JLD
@everywhere using SparseArrays

py"""
import scipy.sparse as sp

def spload(x):
    return sp.load_npz(x)

def spsave(x):
    return sp.save_npz("Results/results.npz", x)
"""

PyObject(S::SparseMatrixCSC) =
    pyimport("scipy.sparse").csc_matrix((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))

# G = py"spload"("Data/Gcsc.npz")
# G = SparseMatrixCSC(G.shape[1], G.shape[2], G.indptr .+ 1, G.indices .+ 1, G.data)

G = npzread("Data/G.npy")

@everywhere G = $G

@everywhere function fit_model(G,dv,i)
    if isfile("Results/Outputs/output$i.jld") == false
        dn = std(dv)
        dv ./= dn
        path = fit(LassoPath, G, dv, α=0.95, λminratio = 1e-4, standardize=false, intercept=false) 
        #λmin = path.λ[argmin(aicc(path))]
        # m = coef(path; select=MinAICc())
        m = coef(path; select=MinCVmse(path))
        #save("Results/Outputs/output$i.jld", Dict("i"=>i,"reg"=>λmin, "dn"=>dn, "m"=>m))
        save("Results/Outputs/output$i.jld", Dict("i"=>i, "dn"=>dn, "m"=>m))
    end
end

@everywhere fit_function(dv,i) = fit_model(G,dv,i)

d = npzread("Data/wvtdata.npy")
d = [(i, d[i,:]) for i = 1:size(d,1)];

p = Progress(length(d), dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
progress_pmap(d, progress=p) do x
    fit_function(x[2], x[1])
end

results = [load("Results/Outputs/output$i.jld") for i = 1:size(d,1)]

sres = reduce(hcat, [x["m"].*x["dn"] for x in results])
py"spsave"(PyObject(sres))

# save("output.jld", "results", results)
