using JuMP
using Ipopt
using LinearAlgebra


X = randn(10,1000)
Xn = sqrt.(sum(X.^2, dims=2)) 
X ./= Xn
ϵ = 1e-1
λ = 1.0
α = 1.0
n_eigs = 5
v, V = eigen(X'*X + ϵ*I)
Γ = V[:, (end-n_eigs+1):end]

μ = rand(10)
U = copy(Γ)

function Ml(μ, X, ϵ)
    M = ϵ*I + X'*Diagonal(μ)*X
    Mc = cholesky(Hermitian(M))
    return (M, Mc)
end

Ml(μ) = Ml(μ, X, ϵ)

_, Mc = Ml(μ)

f(U) = α*sum((U.-Γ).^2) + λ*tr(U'*(Mc\U))

function g!(gvec, U, m, n)
    for i = 1:n 
        gvec[((i-1)*m+1):i*m] .= 2*α*(U[1:m, i].-Γ[1:m, i]) .+ 2*λ*(Mc\U[1:m, i]) 
    end
end

m, n = size(U)

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

model = Model(optimizer_with_attributes(Ipopt.Optimizer))
JuMP.register(model, :f, m*n, jump_wrapper_f, jump_wrapper_g!)
JuMP.register(model, :sumprod, 2*m, sumprod, sumprod_grad!)

@variable(model, x[1:m, 1:n])

set_start_value.(x, U)

# This is the problem section because of the indexing expressions being splatted

for i = 1:n
    for j = i:n
        x_i = x[1:m, i]
        x_j = x[1:m, j]
        @NLconstraint(model, sumprod(x_i..., x_j...) == ((i==j) ? 1 : 0))
    end
end

@NLobjective(model, Min, f(x...))

optimize!(model)
