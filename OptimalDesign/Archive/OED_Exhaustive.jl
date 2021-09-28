using NPZ
using LinearAlgebra
using Combinatorics
using ProgressBars
using Printf: @sprintf
using OnlineStats


G1 = npzread("Kernels/G1.npy")
G2 = npzread("Kernels/G2.npy")
G3 = npzread("Kernels/G3.npy")


function d_oed_exhaust(X1::Matrix{T}, X2::Matrix{T}, X3::Matrix{T}, B, ϵ, λ) where T <: Number
    X = [X3 zeros(T, size(X3));
         zeros(T, size(X3)) X3;
         X1 X2]
    # Xn = maximum(sqrt.(sum(X.^2, dims=2)))
    Xn = sqrt.(sum(X.^2, dims=2)) #normalize all rows to be 1?
    X ./= Xn

    n = size(X3, 1)
    xdas = ones(T, size(X1,1))
    
    function Ml(μ, X, ϵ)
        M = ϵ*I + X'*(Diagonal(μ)*X)
        Mc = cholesky(Hermitian(M))
        l = μ.^2 .*diag(X*(Mc\X'))
        return (M, Mc, l)
    end

    Ml(μ) = Ml(μ, X, ϵ)
    
    function f(μ, λ)
        M, Mc, l = Ml(μ) 
        return -logdet(M) + λ*maximum(l)
    end

    f(μ) = f(μ, λ)

    wrap_f(x) = f(vcat(x, x, xdas))

    combs = combinations(1:n,B)

    f_vals = zeros(length(combs))
    x = zeros(n, length(combs))

    iter = ProgressBar(enumerate(combs))
    stat = Extrema()
    for (i, comb) in iter
        x[comb,i] .= 1
        f_vals[i] = wrap_f(x[:,i])
        fit!(stat, f_vals[i])
        set_description(iter, "$(values(stat))")
    end
    
    return (f_vals, x)
end

f_vals, x = d_oed_exhaust(G1, G2, G3, 4, 1e-1, 10.0)

npzwrite("res2.npy", x[:,argmin(f_vals)])
# ;
