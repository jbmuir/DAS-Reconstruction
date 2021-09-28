using NPZ
using JuMP
using Ipopt
using LinearAlgebra
using WoodburyMatrices
using .Threads
using ProgressMeter
import Base: isless

struct TestResult
    i::Int64
    f::Float64
end
isless(a::TestResult, b::TestResult) = isless(a.f, b.f)

function main()
    T = Float64
    X1 = npzread("Kernels/G1.npy")
    X2 = npzread("Kernels/G2.npy")
    X3 = npzread("Kernels/G4.npy")
    nfactor = npzread("Testing/nfactor.npy")

    ϵ = 1e-2
    λ = 1.0
    lnorm = -log(ϵ)*(2*size(X1,2))

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
        l2 = l[1:n] .+ l[(n+1):2*n] # coherence of point sensors only; sum across channels
        return -logdet(M) + λ*maximum(l2) - lnorm
    end

    f(μ) = f(μ, λ, lnorm, n)
    
    function testf(u, i)
        u_test = copy(u)
        u_test[i] = 1.0
        u_test[i+64] = 1.0
        return TestResult(i, f(u_test))
    end
    
    u = vcat(zeros(64+64), ones(size(X1,1)))
    to_test = Set(1:64)
    selection_order = Int[]
    f_vals = T[]
    p = Progress(32*65, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    while length(to_test) > 0
        test_vec = TestResult[]
        for i in collect(to_test)
            push!(test_vec, testf(u, i))
            next!(p)
        end
        best = minimum(test_vec)
        delete!(to_test, best.i)
        push!(selection_order, best.i)
        push!(f_vals, best.f)
        u[best.i] = 1.0
        u[best.i+64] = 1.0
    end

    npzwrite("IncoherenceResults/seq_order.npy", selection_order)
    npzwrite("IncoherenceResults/seq_vals.npy", f_vals)
    
end


main()
# npzwrite("res_mixed.npy", value.(x))
# ;
