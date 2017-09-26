# FiniteDiffCheck

[![Build Status](https://travis-ci.org/magerton/FiniteDiffCheck.jl.svg?branch=master)](https://travis-ci.org/magerton/FiniteDiffCheck.jl)

[![Coverage Status](https://coveralls.io/repos/magerton/FiniteDiffCheck.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/magerton/FiniteDiffCheck.jl?branch=master)

[![codecov.io](http://codecov.io/github/magerton/FiniteDiffCheck.jl/coverage.svg?branch=master)](http://codecov.io/github/magerton/FiniteDiffCheck.jl?branch=master)


Checks analytic vs finite difference gradient + hessian using central differencing.
While the [Calculus.jl](https://github.com/johnmyleswhite/Calculus.jl) package does
lots more & takes one-sided differences, it won't do Jacobians (though [Calculus2.jl](https://github.com/johnmyleswhite/Calculus2.jl) does). Also, `FiniteDiffCheck.jl` can use `AbstractVector`s like `MVector` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).


# Example:

```julia
using FiniteDiffCheck

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

θ0 = rand(2)

# when defining tmp struct
tmp = jacchecktmp(θ0)
@show out = jac_hess_check!(tmp, fgh!, θ0)
@show all(out .< 3e-10)

# without
@show jac_hess_check(fgh!, θ0)
```
