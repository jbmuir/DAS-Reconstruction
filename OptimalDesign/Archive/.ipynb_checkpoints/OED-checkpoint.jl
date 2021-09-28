using MAT
using Convex
using ECOS
using Plots
using LinearAlgebra
using LineSearches
using ProgressBars
using WoodburyMatrices
using Printf: @sprintf

basis = matread("../Curvelet_Basis_Construction/G_32_32.mat")
G = basis["G_mat"]
s = basis["scales"]
G = G.*(2 .^ (-2 .*s))
Gn = maximum(sqrt.(sum(G.^2, dims=2)))
G ./= Gn

function d_oed(X::Matrix{T}, B, ϵ, λ; maxiter=2500, ls=MoreThuente(), g_rtol::T = sqrt(eps(T)), g_atol::T = eps(T)) where T <: Number
    #initial vector
    n = size(X, 1)
    μ = ones(n, 1)
    μp = Variable(n, 1)
    gvec = similar(μ)
    s = similar(μ)
    #setup linesearch functions
    #linesearch - this needs to make sure that any interior functions actually have function barriers around global variables
    
    function Ml(μ, X, ϵ)
        println(Diagonal(μ))
        M = ϵ*I + X'*Diagonal(μ)*X
        Mc = cholesky(Hermitian(M))
        l = μ.*diag(X*(Mc\X'))
        return (M, Mc, l)
    end

    Ml(μ) = Ml(μ, X, ϵ)
    
    function f(μ, λ)
        M, Mc, l = Ml(μ) 
        return -logdet(M) + λ*maximum(l)
    end

    f(μ) = f(μ, λ)
    
    function gi!(gvec, i, Mc, μ, u, lmind, X, λ, T)
        ld_term = @views -X[i,:]'*(Mc\X[i,:])
        tmp_A = SymWoodbury(Mc, X[i,:], -μ[i])
        tmp_b = @views tmp_A \ X[i,:]
        tmp_1 = (u'*tmp_b)^2
        tmp_d = @views X[i,:]'*tmp_b
        tmp_den = (1+μ[i]*tmp_d)^2
        lev_term_1 =  -μ[lmind]*((1+μ[i]*tmp_d)*tmp_1 - μ[i]*tmp_1*tmp_d) / tmp_den
        lev_term_2 = (i==lmind) ? u'*(Mc\u) : zero(T)
        gvec[i] = ld_term + λ*(lev_term_1 + lev_term_2)
    end
    
    gi!(gvec, i, Mc, μ, u, lmind) = gi!(gvec, i, Mc, μ, u, lmind, X, λ, T)

    function g!(gvec, μ)
        M, Mc, l = Ml(μ) 
        lm, lmind = findmax(l)
        lmind=lmind[1]
        u = @views X[lmind, :]
        for i = 1:n
            gi!(gvec, i, Mc, μ, u, lmind)
        end
    end

    function fg!(gvec, μ)
        M, Mc, l = Ml(μ) 
        lm, lmind = findmax(l)
        lmind=lmind[1]
        u = @views X[lmind, :]
        for i = 1:n
            gi!(gvec, i, Mc, μ, u, lmind)
        end
        return -logdet(M) + λ*maximum(l)
    end
    
    ϕ(α) = f(μ.+α.*s)
    
    function dϕ(α)
        g!(gvec, μ .+ α.*s)
        return dot(gvec, s) 
    end
    
    function ϕdϕ(α)
        phi = fg!(gvec, μ .+ α.*s)
        dphi = dot(gvec, s)
        return (phi, dphi)
    end

    #get initial gradient and iterate
    g!(gvec, μ)
    fμ = f(μ)
    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)    
    iter = ProgressBar(1:maxiter)
    set_description(iter, string(@sprintf("Starting Up")))

    for i in iter
        if gnorm < gtol
            break
        else
            #set step direction
            s .= -gvec
            dϕ_0 = dot(s, gvec)
            #linesearch gradient
            α, fμ = ls(ϕ, dϕ, ϕdϕ, 1.0, fμ, dϕ_0) 
            # α = 0.001
            # fμ = f(μ)
            @. μ = μ + α*s    
            #project into constrained space - decrease budget as we go along

            projection = minimize(sumsquares(μp-μ), [0<=μp, μp<=1, sum(μp)<=max(n-i, B)])
            solve!(projection, () -> ECOS.Optimizer(verbose=false))   
            μ .= μp.value
            #update gradient
            g!(gvec, μ)
            gnorm = norm(gvec)
            
            set_description(iter, string(@sprintf("Objective: %.5f", fμ)))
        end
    end
    
    return μ, round.(μ)
end

resuround, res = d_oed(G, 23, 1.0, 10.0, maxiter=10)


