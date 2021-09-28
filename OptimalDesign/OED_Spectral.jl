using NPZ
using JuMP
using Ipopt
using LinearAlgebra
using .Threads
using WoodburyMatrices
using Statistics
using UnicodePlots


function main(B)
    X1 = npzread("Kernels/G1.npy")
    X2 = npzread("Kernels/G2.npy")
    X3 = npzread("Kernels/G3.npy")
    nfactor = npzread("Testing/nfactor.npy")

    n = size(X3, 1)
    ndas = size(X1, 1)

    T = Float64
    ϵ = 1.0
    λ = 100.0
    α = 100.0

    X = [X3 zeros(T, size(X3));
        zeros(T, size(X3)) X3;
        nfactor*X1 nfactor*X2]
    Xn = maximum(sqrt.(sum(X.^2, dims=2)))
    #Xn = sqrt.(sum(X.^2, dims=2)) #normalize all rows to be 1?
    X ./= Xn

    v, V = eigen(X'*X + ϵ*I)

    #eigtol = 5#quantile(v, 0.99)

    n_eigs = 5#sum(v.>=eigtol)

    Γ = V[:, (end-n_eigs+1):end]

    μ_init = ones(n) ./ n .* B
    μn = sum(μ_init)/B
    μ_init ./= μn
    U_init = copy(Γ)

    function Ml(μ, X, ϵ)
        M = ϵ*I + X'*Diagonal(μ)*X
        Mc = cholesky(Hermitian(M))
        return (M, Mc)
    end

    Ml(μ) = Ml(μ, X, ϵ)

    function subproblem_sμ(μ_init, u, X, λ, B, maxiter=1000)

        function f(μ, u, λ)
            M, Mc = Ml(μ) 
            return -logdet(M) + λ*tr(u'*(Mc\u))
        end

        f(μ) = f(μ, u, λ) 

        jump_wrapper_f(x...) = f(vcat([x[i] for i in 1:length(x)], [x[i] for i in 1:length(x)], ones(T, ndas)))

        function gi(i, Mc, μ, u, λ)
            ld_term = @views -X[i,:]'*(Mc\X[i,:])
            tmp_A = SymWoodbury(Mc, X[i,:], -μ[i])
            tmp_b = @views tmp_A \ X[i,:]
            #tmp_1 = (u'*tmp_b)^2
            tmp_1 = sum((u'*tmp_b).^2) # think that this should be correct for tr(U'MU) for matrix U but need to check
            tmp_d = @views X[i,:]'*tmp_b
            tmp_den = (1+μ[i]*tmp_d)^2
            eig_term = ((1+μ[i]*tmp_d)*tmp_1 - μ[i]*tmp_1*tmp_d) / tmp_den
            return ld_term - λ*eig_term #apparently need a -ve sign here, work out why
        end
        
        function g!(gvec, μ, X, u, λ, n)
            M, Mc = Ml(μ) 
            @threads for i = 1:n
                g1 = gi(i, Mc, μ, u, λ) # in respect to x component
                g2 = gi(i+n, Mc, μ, u, λ) # in respect to y component
                gvec[i] = g1 + g2
            end
        end
        
        g!(gvec, μ) = g!(gvec, μ, X, u, λ, n)
            
        jump_wrapper_g!(gvec, x...) = g!(gvec, vcat([x[i] for i in 1:length(x)], [x[i] for i in 1:length(x)], ones(T, ndas)))

        model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>maxiter, "tol"=>1e-4/B))
        JuMP.register(model, :f, n, jump_wrapper_f, jump_wrapper_g!)

        @variable(model, 0.0<=x[1:n]<= 1.0)
        
        set_start_value.(x, μ_init)

        @NLconstraint(model, +(x...) <= B)
        
        @NLobjective(model, Min, f(x...))

        optimize!(model)

        return (value.(x), vcat(value.(x), value.(x), ones(T, ndas)), objective_value(model))

    end

    function subproblem_sO(u_init, μ, γ, X, λ, α, maxiter=1000)
        _, Mc = Ml(μ)

        f(u) = α*sum((u.-γ).^2) + λ*tr(u'*(Mc\u))
        function g!(gvec, u) 
            gvec .= 2*α*(u-γ) + 2*λ*(Mc\u)
        end
        jump_wrapper_f(x...) = f([x[i] for i in 1:length(x)])
        jump_wrapper_g!(gvec, x...) = g!(gvec, [x[i] for i in 1:length(x)])

        sumsquares(x...) = sum(x[i]^2 for i in 1:length(x))
        function sumsquares_grad!(gvec, x...)
            for i in 1:length(x)
                gvec[i] = 2*x[i]
            end
        end

        model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>maxiter))
        JuMP.register(model, :f, length(γ), jump_wrapper_f, jump_wrapper_g!)
        JuMP.register(model, :sumsquares, length(γ), sumsquares, sumsquares_grad!)

        @variable(model, x[1:length(γ)])

        set_start_value.(x, u_init)

        @NLconstraint(model, sumsquares(x...) == 1)
        
        @NLobjective(model, Min, f(x...))

        optimize!(model)

        return (value.(x), objective_value(model))

    end

    function subproblem_sO(U_init, μ, Γ, X, λ, α, m, n, maxiter=1000)
            _, Mc = Ml(μ)

            f(U) = α*sum((U.-Γ).^2) + λ*tr(U'*(Mc\U))

            function g!(gvec, U, m, n)
                @threads for i = 1:n 
                    @views gvec[((i-1)*m+1):i*m] .= 2*α*(U[1:m, i].-Γ[1:m, i]) .+ 2*λ*(Mc\U[1:m, i]) 
                end
            end

            L = LinearIndices((m, n))
            jump_wrapper_f(x...) = f([x[L[i,j]] for i in 1:m, j in 1:n])
            jump_wrapper_g!(gvec, x...) = g!(gvec, [x[L[i,j]] for i in 1:m, j in 1:n], m, n)

            sumprod(x...) = sum(x[i]*x[i+m] for i in 1:m)
            function sumprod_grad!(gvec, x...)
                for i in 1:m
                    gvec[i] = x[i+m]
                    gvec[i+m] = x[i]
                end
            end

            model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>maxiter))
            JuMP.register(model, :f, m*n, jump_wrapper_f, jump_wrapper_g!)
            JuMP.register(model, :sumprod, 2*m, sumprod, sumprod_grad!)

            @variable(model, x[1:m, 1:n])
        
            set_start_value.(x, U_init)

            for i = 1:n
                for j = i:n
                    x_i = x[1:m, i]
                    x_j = x[1:m, j]
                    @NLconstraint(model, sumprod(x_i..., x_j...) == ((i==j) ? 1 : 0))            end
            end

            @NLobjective(model, Min, f(x...))

            optimize!(model)

            return (value.(x), objective_value(model)) # want to have all of the x here as a matrix
    end

    function optimize_spectral(μ_init, u_init, iters, γ, X, λ, α, B)
        μ = copy(μ_init)
        u = copy(u_init)
        ob1l = Float64[]
        ob2l = Float64[]
        for iter in 1:iters
            μ, μl, ob1 = subproblem_sμ(μ, u, X, λ, B)
            u, ob2 = subproblem_sO(u, μl, γ, X, λ, α)
            push!(ob1l, ob1)
            push!(ob2l, ob2)
        end
        μ, μl, ob1 = subproblem_sμ(μ, u, X, λ, B)
        push!(ob1l, ob1)

        return (μ, u, ob1l, ob2l)
    end

    function optimize_spectral(μ_init, U_init, iters, Γ, X, λ, α, B, m, n)
        μ = copy(μ_init)
        U = copy(U_init)
        ob1l = Float64[]
        ob2l = Float64[]
        for iter in 1:iters
            μ, μl, ob1 = subproblem_sμ(μ, U, X, λ, B)
            U, ob2 = subproblem_sO(U, μl, Γ, X, λ, α, m, n)
            push!(ob1l, ob1)
            push!(ob2l, ob2)
        end
        μ, μl, ob1 = subproblem_sμ(μ, U, X, λ, B)
        push!(ob1l, ob1)

        return (μ, U, ob1l, ob2l)
    end

    μ, U_res, ob1l, ob2l = optimize_spectral(μ_init, U_init, 5, Γ, X, λ, α, B, size(Γ)...)

    M, _ = Ml(vcat(μ,μ, ones(ndas)))
    vμ, Vμ = eigen(M)

    npzwrite("SpectralResults/res_spectral_$B.npy", μ)

    lineplot(ob1l)

    lineplot(ob2l)
end

for i in [4,8,16,32,64,128,256]
    main(i)
end

#yeh