using NPZ
using LinearAlgebra
using .Threads
using Arpack
using ProgressMeter
import Base: isless

struct TestResult
    i::Int64
    f::Float64
end
isless(a::TestResult, b::TestResult) = isless(a.f, b.f)

function main()
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
    X ./= Xn

    n_eigs = 5

    v, V = eigs(X'*X + ϵ*I; nev=n_eigs, which=:LM)

    Γ = V[:, (end-n_eigs+1):end]

    function Ml(μ, X, ϵ)
        M = ϵ*I + X'*Diagonal(μ)*X
        Mc = cholesky(Hermitian(M))
        return (M, Mc)
    end

    Ml(μ) = Ml(μ, X, ϵ)

    function f(μ, Γ, α, λ, n_eigs)
        M, Mc = Ml(μ)
        vu, Vu = eigs(M; nev=5, which=:LM)
        U = Vu[:, (end-n_eigs+1):end]
        return -logdet(M) + λ*tr(U'*(Mc\U)) + α*sum((U.-Γ).^2)
    end

    f(μ) = f(μ, Γ, α, λ, n_eigs)

    function testf(u, i)
        u_test = copy(u)
        u_test[i] = 1.0
        u_test[i+256] = 1.0
        return TestResult(i, f(u_test))
    end

    u = vcat(zeros(256+256), ones(size(X1,1)))
    to_test = Set(1:256)
    selection_order = Int[]
    f_vals = T[]
    p = Progress(128*257, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    while length(to_test) > 0
        test_vec = TestResult[]
        @threads for i in collect(to_test)
            push!(test_vec, testf(u, i))
            next!(p)
        end
        best = minimum(test_vec)
        delete!(to_test, best.i)
        push!(selection_order, best.i)
        push!(f_vals, best.f)
        u[best.i] = 1.0
        u[best.i+256] = 1.0
    end

    npzwrite("seq_order.npy", selection_order)
    npzwrite("seq_vals.npy", f_vals)

end

main()