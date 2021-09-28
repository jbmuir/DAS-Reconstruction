using NPZ
using JuMP
using Ipopt
using LinearAlgebra
using WoodburyMatrices
using .Threads
using Juniper
using Cbc
using Gurobi

X1 = npzread("Kernels/G1.npy")
X2 = npzread("Kernels/G2.npy")
X3 = npzread("Kernels/G3.npy")

lnorm = -log(ϵ)*(2*size(X1,2))


function d_oed(X1::Matrix{T}, X2::Matrix{T}, X3::Matrix{T}, B, ϵ, λ, lnorm; maxiter=1000) where T <: Number
    X = [X3 zeros(T, size(X3));
         zeros(T, size(X3)) X3;
         X1 X2]
    #Xn = maximum(sqrt.(sum(X.^2, dims=2)))
    Xn = sqrt.(sum(X.^2, dims=2)) #normalize all rows to be 1?
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
    
    function f(μ, λ, lnorm)
        M, Mc, l = Ml(μ) 
        return -logdet(M) + λ*maximum(l) - lnorm
    end

    f(μ) = f(μ, λ, lnorm)

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
        lm, lmind = findmax(l)
        lmind=lmind[1]
        u = @views X[lmind, :]
        @threads for i = 1:n
            g1 = gi(i, Mc, μ, u, lmind) # in respect to x component
            g2 = gi(i+n, Mc, μ, u, lmind) # in respect to y component
            gvec[i] = g1+g2
        end
    end
    
    g!(gvec, μ) = g!(gvec, μ, X, n)
    
    jump_wrapper_g!(gvec, x...) = g!(gvec, vcat([x[i] for i in 1:length(x)], [x[i] for i in 1:length(x)], ones(T, ndas)))
    
    # optimizer = Juniper.Optimizer
    # nl_solver= optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>1)
    # mip_solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 1)
    # model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver, "log_levels"=>[:Table, :Info], "time_limit"=>10*3600, "feasibility_pump_time_limit"=>300,"strong_branching_time_limit"=>300, "traverse_strategy"=>:BFS,"branch_strategy"=>:Reliability,
    # "registered_functions" => [
    #     Juniper.register(:f, n, jump_wrapper_f, jump_wrapper_g!)
    # ]))
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>maxiter, "tol"=>1e-2))
    JuMP.register(model, :f, n, jump_wrapper_f, jump_wrapper_g!)

    @variable(model, 0.0<=x[1:n]<= 1.0)
    #@variable(model, x[1:n], binary=true)

    @NLconstraint(model, +(x...) <= B)
     
    @NLobjective(model, Min, f(x...))
        
    optimize!(model)
    
    return (model, x)
end

ϵ = 1e-3
lnorm = -log(ϵ)*(2*size(X1,2))
model, x = d_oed(X1, X2, X3, 32, ϵ, 1/ϵ, lnorm, maxiter=1000)
println("Model Objective = $(objective_value(model))")
npzwrite("res_mixed.npy", value.(x))
# ;
