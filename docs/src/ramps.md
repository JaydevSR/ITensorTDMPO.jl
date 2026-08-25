```@meta
CurrentModule = ITensorTDMPO
```

# Ramps

A [`Ramp`](@ref) separates the **shape** of a sweep from its
**endpoints**. A [`RampShape`](@ref) is a pure profile on the unit
interval — `s: [0,1] → [0,1]` with `s(0) = 0` and `s(1) = 1` — and `Ramp`
maps that profile onto physical time and physical values:

```julia
Ramp(shape, t_start, t_stop, value_start, value_stop)
```

Evaluating `ramp(t)` normalizes the time, applies the profile, and
rescales to the value range:

```julia
τ = (t - t_start) / (t_stop - t_start)         # normalized time
s = shape(τ)                                    # normalized progress (clamped)
value_start + (value_stop - value_start) * s    # physical value
```

Clamping happens inside the shape, so outside `[t_start, t_stop]` the
ramp *holds* at its endpoint value rather than extrapolating — you can
keep evolving past `t_stop` and the parameter stays put.

```julia
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)
ramp(0.0)   # 0.0
ramp(5.0)   # 1.0   (midpoint of a symmetric shape)
ramp(10.0)  # 2.0
ramp(50.0)  # 2.0   (held, not extrapolated)
```

Ramps run downward as readily as upward (`value_start > value_stop`),
and nothing requires a `Ramp` at all — any `f(t)` is a valid driving, as
is a plain number for a constant.

## Shapes

| Shape | `s(τ)` | endpoint slope |
|---|---|---|
| `LinearRamp()` | `τ` | **nonzero** at both ends |
| `PowerLawRamp(p)` | `τᵖ` | zero at start for `p > 1` |
| `ExponentialRamp(k)` | `(e^{kτ} − 1)/(e^k − 1)` | asymmetric; `k = 0` is linear |
| `SineRamp()` | `(1 − cos πτ)/2` | zero at both ends |
| `SineSquaredRamp()` | `sin²(πτ/2)` | identical to `SineRamp` (half-angle identity) |
| `SmoothstepRamp()` | `3τ² − 2τ³` | zero at both ends |
| `SmootherstepRamp()` | `6τ⁵ − 15τ⁴ + 10τ³` | zero slope **and** curvature at both ends |

For adiabatic sweeps this choice is substantive. `LinearRamp` switches
the sweep on discontinuously in `Ḣ`, and that kink is a broadband
perturbation which drives diabatic transitions no matter how slowly you
ramp. Shapes with `s′(0) = s′(1) = 0` turn the drive on and off smoothly
and suppress those endpoint excitations; `SmootherstepRamp` additionally
kills the curvature. `PowerLawRamp` and `ExponentialRamp` are the
asymmetric options — useful when you want to move slowly through a gap
minimum at one end only.

## Reference

```@docs
RampShape
Ramp
LinearRamp
SmoothstepRamp
SmootherstepRamp
SineRamp
SineSquaredRamp
PowerLawRamp
ExponentialRamp
```
