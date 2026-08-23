"""
Ramp shapes map a normalized time `τ ∈ [0, 1]` to a normalized progress
`s ∈ [0, 1]`, with `s(0) == 0` and `s(1) == 1`. A [`Ramp`](@ref) combines a
shape with a time window and a value range to produce a function of
physical time, e.g. for use as a coefficient in a time-dependent
Hamiltonian term.
"""
abstract type RampShape end

"""
    LinearRamp()

`s(τ) = τ`.
"""
struct LinearRamp <: RampShape end
(::LinearRamp)(τ) = clamp(τ, 0, 1)

"""
    SmoothstepRamp()

Cubic smoothstep, `s(τ) = 3τ² - 2τ³`. Zero slope at both endpoints.
"""
struct SmoothstepRamp <: RampShape end
function (::SmoothstepRamp)(τ)
    τ = clamp(τ, 0, 1)
    return τ^2 * (3 - 2τ)
end

"""
    SmootherstepRamp()

Quintic smootherstep, `s(τ) = 6τ⁵ - 15τ⁴ + 10τ³`. Zero slope and
curvature at both endpoints.
"""
struct SmootherstepRamp <: RampShape end
function (::SmootherstepRamp)(τ)
    τ = clamp(τ, 0, 1)
    return τ^3 * (τ * (6τ - 15) + 10)
end

"""
    SineRamp()

Raised-cosine ramp, `s(τ) = (1 - cos(πτ)) / 2`. Zero slope at both
endpoints.
"""
struct SineRamp <: RampShape end
function (::SineRamp)(τ)
    τ = clamp(τ, 0, 1)
    return (1 - cospi(τ)) / 2
end

"""
    SineSquaredRamp()

`s(τ) = sin²(πτ/2)`. Zero slope at both endpoints.

By the half-angle identity this is exactly equal to [`SineRamp`](@ref);
both names are provided since either convention is common in the
literature.
"""
struct SineSquaredRamp <: RampShape end
function (::SineSquaredRamp)(τ)
    τ = clamp(τ, 0, 1)
    return sinpi(τ / 2)^2
end

"""
    PowerLawRamp(exponent)

`s(τ) = τ^exponent`. `exponent = 1` is equivalent to [`LinearRamp`](@ref);
`exponent > 1` starts slow and accelerates, `exponent < 1` starts fast and
decelerates.
"""
struct PowerLawRamp <: RampShape
    exponent::Float64
end
function (shape::PowerLawRamp)(τ)
    return clamp(τ, 0, 1)^shape.exponent
end

"""
    ExponentialRamp(rate)

Exponential ramp normalized to `s(0) == 0` and `s(1) == 1`:
`s(τ) = (exp(rate * τ) - 1) / (exp(rate) - 1)`. `rate > 0` starts slow and
accelerates, `rate < 0` starts fast and decelerates, and `rate == 0`
reduces to [`LinearRamp`](@ref).
"""
struct ExponentialRamp <: RampShape
    rate::Float64
end
function (shape::ExponentialRamp)(τ)
    τ = clamp(τ, 0, 1)
    k = shape.rate
    iszero(k) && return τ
    return expm1(k * τ) / expm1(k)
end

"""
    Ramp(shape::RampShape, t_start, t_stop, value_start, value_stop)

A scalar ramp that follows the normalized `shape` from `value_start` at
`t_start` to `value_stop` at `t_stop`. Calling `ramp(t)` gives the ramp
value at time `t`, clamped to `value_start`/`value_stop` outside
`[t_start, t_stop]`.

# Examples

```julia
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 1.0)
ramp(0.0)  # 0.0
ramp(5.0)  # 0.5
ramp(10.0) # 1.0
```
"""
struct Ramp{Shape <: RampShape, T}
    shape::Shape
    t_start::T
    t_stop::T
    value_start::T
    value_stop::T
end

function Ramp(shape::RampShape, t_start, t_stop, value_start, value_stop)
    t_start, t_stop, value_start, value_stop = promote(
        float(t_start), float(t_stop), float(value_start), float(value_stop)
    )
    return Ramp(shape, t_start, t_stop, value_start, value_stop)
end

function (ramp::Ramp)(t)
    τ = (t - ramp.t_start) / (ramp.t_stop - ramp.t_start)
    s = ramp.shape(τ)
    return ramp.value_start + (ramp.value_stop - ramp.value_start) * s
end
