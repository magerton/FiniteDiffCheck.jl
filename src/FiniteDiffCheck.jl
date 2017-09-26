module FiniteDiffCheck

export jacchecktmp, jac_hess_check, jac_hess_check!

struct jacchecktmp{T<:Real,V}
    θ1::V
    θ2::V
    grad1::Vector{T}
    grad2::Vector{T}
    fd_grad::Vector{T}
    ana_grad::Vector{T}
    fd_hess::Matrix{T}
    ana_hess::Matrix{T}
end

function jacchecktmp(θ::AbstractVector{T}) where {T}
    k = length(θ)
    return jacchecktmp{T,typeof(θ)}(similar(θ), similar(θ), Vector{T}(k), Vector{T}(k), Vector{T}(k), Vector{T}(k), Matrix{T}(k,k), Matrix{T}(k,k))
end
# ---------------------------- jacobian check -----------------------------

reldiff(x::T, y::T) where {T<:Real} = x+y == zero(T) ? zero(T) : convert(T,2) * abs(x-y) / (abs(x)+abs(y))
absdiff(x::T, y::T) where {T<:Real} = abs(x-y)

function max_and_pos(diff_fun::Function, x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Real}
    d, ind = findmax(diff_fun.(x, y))
    pos = ind2sub(size(x), ind)
    return (d, pos)
end

"""
    jac_hess_check!(tmp::jacchecktmp, fun::Function, θ::AbstractVector; dtol::Real=1e-7, d2tol::Real=1e-5, warning::Bool=true)

Checks analytic gradients & hessians produced by `fun` against finite difference.
The function must be of the form `f(θ::AbstractVector, du::Vector, d2u::Matrix)`
and update `du` and `d2u` in place. Warns if absolute or relative differences
outside of `dtol` (for gradient) or `d2tol` (for hessian) and `warning=true`.
"""
function jac_hess_check!(tmp::jacchecktmp, fun::Function, θ::AbstractVector{T}; dtol::Real=1e-7, d2tol::Real=1e-5, warning::Bool=true) where {T}
    for k in 1:length(θ)
        tmp.θ1 .= θ
        tmp.θ2 .= θ
        h = max( abs(θ[k]), one(T) ) * cbrt(eps(T))
        tmp.θ1[k] -= h
        tmp.θ2[k] += h

        # Use this instead of 2.0*h to further reduce numerical error.
        # See Miranda & Fackler (2002) pp 103-104
        hh = tmp.θ2[k] - tmp.θ1[k]

        LL1 = fun(tmp.θ1, tmp.grad1, zeros(T,0,0))
        LL2 = fun(tmp.θ2, tmp.grad2, zeros(T,0,0))

        tmp.fd_grad[k]   .=  (LL2 - LL1) ./ hh
        tmp.fd_hess[k,:] .=  (tmp.grad2 .- tmp.grad1) ./ hh
    end

    fun(θ, tmp.ana_grad, tmp.ana_hess)

    g_abs, g_abs_pos = max_and_pos(absdiff, tmp.ana_grad, tmp.fd_grad)
    g_rel, g_rel_pos = max_and_pos(reldiff, tmp.ana_grad, tmp.fd_grad)

    h_abs, h_abs_pos = max_and_pos(absdiff, tmp.ana_hess, tmp.fd_hess)
    h_rel, h_rel_pos = max_and_pos(reldiff, tmp.ana_hess, tmp.fd_hess)

    if warning == true
        g_abs < dtol  || warn("For g(x), max abs diff: $g_abs @ $g_abs_pos,  max rel diff: $g_rel @ $g_rel_pos")
        h_abs < d2tol || warn("For h(x), max abs diff: $h_abs @ $h_abs_pos,  max rel diff: $h_rel @ $h_rel_pos")
    end

    return (g_abs, h_abs)
end


jac_hess_check(fun::Function, θ::AbstractVector; kwargs...) = jac_hess_check!(jacchecktmp(θ), fun, θ; kwargs...)


# module end
end
