using FiniteDiffCheck
using Base.Test

function fgh!(x::AbstractVector, dx::AbstractVector, d2x::AbstractMatrix)
    length(x) == 2 || throw(DimensionMismatch())

    f = x[1]^2. + 3.*x[1]*x[2] + x[2]^2.

    if length(dx) > 0
        dx[1] = 2.*x[1] + 3.*x[2]
        dx[2] = 3.*x[1] + 2.*x[2]
    end

    if length(d2x) > 0
        d2x[1,1] = 2.
        d2x[2,1] = 3.
        d2x[1,2] = 3.
        d2x[2,2] = 2.
    end

    return f
end


function fgh2!(x::AbstractVector, dx::AbstractVector, d2x::AbstractMatrix)
    length(x) == 2 || throw(DimensionMismatch())

    f = x[1]^2. + 3.*x[1]*x[2] + x[2]^2.

    if length(dx) > 0
        dx[1] = 2.*x[1] + 3.*x[2]
        dx[2] = 3.*x[1] + 2.*x[2]
    end

    if length(d2x) > 0
        d2x[1,1] = 2.
        d2x[2,1] = 3.
        d2x[1,2] = 3.
        d2x[2,2] = 2.
        d2x .*= 2.0
    end

    return f
end


srand(1234)

@show θ0 = rand(2)
tmp = jacchecktmp(θ0)

@show out = jac_hess_check!(tmp, fgh!, θ0)
@test all(out .< 3e-10)


@show out2 = jac_hess_check(fgh2!, θ0)
@test !all(out2 .< 3e-10)


















#
