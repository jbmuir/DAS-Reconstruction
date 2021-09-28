using NPZ
using Convex
using ECOS
using LinearAlgebra
using LineSearches
using ProgressBars
using WoodburyMatrices
using Printf: @sprintf
using .Threads

G1 = npzread("Kernels/G1.npy")
G2 = npzread("Kernels/G2.npy")
G3 = npzread("Kernels/G3.npy")
nfactor = npzread("Testing/nfactor.npy")

ϵ = 1e-1
lnorm = -log(ϵ)*(2*size(G1,2))


function d_oed(X1::Matrix{T}, X2::Matrix{T}, X3::Matrix{T}, nfactor, B, ϵ, λ, lnorm; maxiter=2500, ls=MoreThuente()) where T <: Number
    #initial vector
    X = [X3 zeros(T, size(X3));
         zeros(T, size(X3)) X3;
         nfactor*X1 nfactor*X2]
    Xn = maximum(sqrt.(sum(X.^2, dims=2)))
    X ./= Xn

    n = size(X3, 1)
    na = size(X, 1)
    μr = ones(T, n)./n
    μr_tmp = zeros(T, n)
    μp = Variable(n)
    μ = ones(T, na) 
    μ[1:n] .= μr
    μ[(n+1):(2*n)] .= μr
    gvec = zeros(T, size(μ))
    s = zeros(T, size(μ))   
    
    function project!(μ, μr, μp, B)
            μr .= μ[1:n]
            projection = minimize(sumsquares(μp-μr), [0<=μp, μp<=1, sum(μp)<=B])
            solve!(projection, () -> ECOS.Optimizer(verbose=false))   
            μr .= @views μp.value[:,1]
            μ[1:n] .= μr
            μ[(n+1):(2*n)] .= μr
            return μ
    end
    
    project!(μ) = project!(μ, μr_tmp, μp, B)
    
    #setup linesearch functions
    #linesearch - this needs to make sure that any interior functions actually have function barriers around global variables
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
    
    function ff(μ, X, ϵ, λ, n)
        M = ϵ*I + X'*Diagonal(μ)*X
        l = (μ.^2).*diag(X*(M\X'))
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        return -logdet(M) + λ*maximum(l2)
    end

    ff(μ) = ff(μ, X, ϵ, λ, n)
    
    function gi(i, Mc, μ, u, lmind, X, λ, T)
        ld_term = @views -X[i,:]'*(Mc\X[i,:])
        tmp_A = SymWoodbury(Mc, X[i,:], -μ[i])
        tmp_b = @views tmp_A \ X[i,:]
        tmp_1 = (u'*tmp_b)^2
        tmp_d = @views X[i,:]'*tmp_b
        tmp_den = (1+μ[i]*tmp_d)^2
        lev_term_1 =  -μ[lmind]*((1+μ[i]*tmp_d)*tmp_1 - μ[i]*tmp_1*tmp_d) / tmp_den
        lev_term_2 = (i==lmind) ? u'*(Mc\u) : zero(T)
        return ld_term + λ*(lev_term_1 + lev_term_2)
    end
    
    gi(i, Mc, μ, u, lmind) = gi(i, Mc, μ, u, lmind, X, λ, T)

    function g!(gvec, μ, X, n)
        M, Mc, l = Ml(μ) 
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        lm, lmind = findmax(l2)
        u1 = @views X[lmind, :]
        u2 = @views X[lmind+n, :]
        @threads for i = 1:n
            g1 = gi(i, Mc, μ, u1, lmind) # in respect to x component
            g2 = gi(i+n, Mc, μ, u2, lmind+n) # in respect to y component
            gvec[i] = g1+g2
            gvec[i+n] = g1+g2
        end
    end
    
    g!(gvec, μ) = g!(gvec, μ, X, n)
    
        
    function fg!(gvec, μ, X, n)
        M, Mc, l = Ml(μ) 
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        lm, lmind = findmax(l2)
        u1 = @views X[lmind, :]
        u2 = @views X[lmind+n, :]
        @threads for i = 1:n
            g1 = gi(i, Mc, μ, u1, lmind) # in respect to x component
            g2 = gi(i+n, Mc, μ, u2, lmind+n) # in respect to y component
            gvec[i] = g1+g2
            gvec[i+n] = g1+g2
        end
        return -logdet(M) + λ*maximum(l)
    end
    
    fg!(gvec, μ) = fg!(gvec, μ, X, n)
    
    ϕ(α) = f(project!(μ.+α.*s))
    
    function dϕ(α)
        g!(gvec, project!(μ.+α.*s))
        return dot(gvec, s) 
    end
    
    function ϕdϕ(α)
        phi = fg!(gvec, project!(μ.+α.*s))
        dphi = dot(gvec, s)
        return (phi, dphi)
    end

    #get initial gradient and iterate
    g!(gvec, μ)

    fμ = f(μ)
    fμt = []
    μrarr = []
    push!(fμt, fμ) 
    iter = ProgressBar(1:maxiter)
    set_description(iter, string(@sprintf("Starting Up")))

    for i in iter
        #set step direction
        s .= -gvec
        dϕ_0 = dot(s, gvec)
        #linesearch gradient
        α, fμ = ls(ϕ, dϕ, ϕdϕ, 1.0, fμ, dϕ_0) 
        @. μ = μ + α*s
        project!(μ, μr, μp, B)
        fμ = f(μ)
        #update gradient
        g!(gvec, μ)
        set_description(iter, string(@sprintf("Objective: %.5f Update: %.2e", fμ, fμ-fμt[i])))
        push!(fμt, fμ)
        push!(μrarr, copy(μr))
    end

    function pround(μ, μr, n, B)
        idx = 1:length(μr)
        rμr = copy(μr)
        v1 = copy(μr)
        v2 = copy(μr)
        μ1 = copy(μ)
        μ2 = copy(μ)
        not_zero_or_one(x) = !(isapprox(x,0) || isapprox(x,1))
        feasible(v) = (all(v.>=0) && all(v.<=1) && (sum(v)<=B))
        while length(idx[not_zero_or_one.(rμr)]) > 1
            # j, k = rand(idx[not_zero_or_one.(rμr)], 2)
            idr = idx[not_zero_or_one.(rμr)]
            j = idr[argmax(rμr[idr])]
            k = idr[argmin(rμr[idr])]
            println(j, " ", k)
            δ = min(1-rμr[j], rμr[k])
            τ = min(1-rμr[k], rμr[j])
            v1[j] += δ
            v1[k] -= δ
            v2[j] -= τ
            v2[k] += τ
            sv1 = sum(v1)
            sv2 = sum(v2)
            if !feasible(v1) && !feasible(v2)
                error("No Feasible solution")
            elseif !feasible(v1)
                rμr .= v2
                v1 .= v2
            elseif !feasible(v2)
                rμr .= v1
                v2 .= v1
            else
                μ1[1:n] .= v1
                μ1[(n+1):(2*n)] .= v1
                f1 = ff(μ1)
                μ2[1:n] .= v2
                μ2[(n+1):(2*n)] .= v2
                f2 = ff(μ2)
                if f1 < f2
                    rμr .= v1
                else
                    rμr .= v2
                end
            end
        end
        μ1[1:n] .= rμr
        μ1[(n+1):(2*n)] .= rμr
        fr = ff(μ1)
        return (rμr, fr)
    end

    rμr, fr = pround(μ, μr, n, B)

    return μr, fμt, μrarr, Xn, rμr, fr
end

μr, fμt, μrarr, Xn, rμr,  fr = d_oed(G1, G2, G3, nfactor, 32, ϵ, 1/ϵ, lnorm, maxiter=250)

npzwrite("resmin.npy", μr)
npzwrite("resround.npy", rμr)
;
