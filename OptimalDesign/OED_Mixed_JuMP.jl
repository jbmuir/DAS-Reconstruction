using NPZ
using JuMP
using Ipopt
using LinearAlgebra
using WoodburyMatrices
using .Threads
using ProgressMeter



X1 = npzread("Kernels/G1.npy")
X2 = npzread("Kernels/G2.npy")
X3 = npzread("Kernels/G4.npy") #try on restricted grid
nfactor = npzread("Testing/nfactor.npy")

ϵ = 1e-2
lnorm = -log(ϵ)*(2*size(X1,2))

function d_oed(X1::Matrix{T}, X2::Matrix{T}, X3::Matrix{T}, nfactor, B, ϵ, λ, lnorm; maxiter=1000) where T <: Number
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
        l2 = l[1:2*n] # coherence of point sensors only
        return -logdet(M) + λ*maximum(l2) - lnorm
    end

    f(μ) = f(μ, λ, lnorm, n)

    grad_f(y) = f(vcat([y[i] for i in 1:length(y)], [y[i] for i in 1:length(y)], ones(T, ndas)))

    jump_wrapper_f(x...) = f(vcat([x[i] for i in 1:length(x)], [x[i] for i in 1:length(x)], ones(T, ndas)))
    
    function gi(i, Mc, μ, u, lmind, X, λ, T)
        ld_term = @views -X[i,:]'*(Mc\X[i,:])
        tmp_A = SymWoodbury(Mc, X[i,:], -μ[i])
        tmp_b = @views tmp_A \ X[i,:]
        tmp_1 = (u'*tmp_b)^2
        tmp_d = @views X[i,:]'*tmp_b
        tmp_den = (1+μ[i]*tmp_d)^2
        lev_term_1 = -μ[lmind]^2*((1+μ[i]*tmp_d)*tmp_1 - μ[i]*tmp_1*tmp_d) / tmp_den
        lev_term_2 = (i==lmind) ? 2*μ[lmind]*(u'*(Mc\u)) : zero(T)
        return ld_term + λ*(lev_term_1 + lev_term_2)
    end
    
    gi(i, Mc, μ, u, lmind) = gi(i, Mc, μ, u, lmind, X, λ, T)

    function g!(gvec, μ, X, n)
        M, Mc, l = Ml(μ) 
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        lm, lmind = findmax(l2)
        u1 = @views X[lmind, :]
        u2 = @views X[lmind, :]
        @threads for i = 1:n
            g1 = gi(i, Mc, μ, u1, lmind) # in respect to x component
            g2 = gi(i+n, Mc, μ, u2, lmind) # in respect to y component
            gvec[i] = g1+g2
        end
    end
    
    g!(gvec, μ) = g!(gvec, μ, X, n)
    
    jump_wrapper_g!(gvec, x...) = g!(gvec, vcat([x[i] for i in 1:length(x)], [x[i] for i in 1:length(x)], ones(T, ndas)))

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>maxiter, "tol"=>1e-3))
    JuMP.register(model, :f, n, jump_wrapper_f, jump_wrapper_g!)

    @variable(model, 0.0<=x[1:n]<= 1.0)

    @NLconstraint(model, +(x...) <= B)
     
    @NLobjective(model, Min, f(x...))
        
    optimize!(model)
    
    function ff(μ, X, ϵ, λ, n)
        M = ϵ*I + X'*Diagonal(μ)*X
        l = (μ.^2).*diag(X*(M\X'))
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        return -logdet(M) + λ*maximum(l2) - lnorm
    end
    
    ff(μ) = ff(μ, X, ϵ, λ, n)
    
    μr = clamp.(value.(x), 0, 1) # because it probably hasn't completely met the constraints, make sure it is feasible.
    μ = vcat(μr, μr, ones(T, ndas))

    function pround(μ, μr, n, B)
        p = Progress(n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
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
            next!(p)
        end
        μ1[1:n] .= rμr
        μ1[(n+1):(2*n)] .= rμr
        fr = ff(μ1)
        return (rμr, fr)
    end

    #rμr, fr = pround(μ, μr, n, B)
    
    npzwrite("IncoherenceResults/res_inc_$B.npy", value.(x))
    #npzwrite("IncoherenceResults/res_inc_$(B)_round.npy", rμr)

    return (model, x)#, rμr, fr)
end

for i in [4,8,16,32,64]
    # model, x, rμr, fr = d_oed(X1, X2, X3, nfactor, i, ϵ, 1/ϵ, lnorm, maxiter=500)
#     model, x = d_oed(X1, X2, X3, nfactor, i, ϵ, 1/ϵ, lnorm, maxiter=500)
    model, x = d_oed(X1, X2, X3, nfactor, i, ϵ, 1.0, lnorm, maxiter=50)
end


# model, x, rμr, fr = d_oed(X1, X2, X3, nfactor, 32, ϵ, 1/ϵ, lnorm, maxiter=2)
# println("Model Objective = $(objective_value(model))")
# # npzwrite("res_mixed.npy", value.(x))
# # ;
