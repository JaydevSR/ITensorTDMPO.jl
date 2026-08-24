```@meta
CurrentModule = TDVPlus
```

# Piecewise-constant TDVP

The simplest algorithm: freeze `H(t)` once per interval and run an
ordinary TDVP sweep across it.

```julia
using ITensorMPS, TDVPlus

sites = siteinds("S=1/2", 20)
Hzz = MPO(OpSum() + ..., sites)          # coupling
Hx  = MPO(OpSum() + ..., sites)          # transverse field
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

ψ = time_evolve(
    [(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
    alg = "piecewise_constant", nsteps = 200, cutoff = 1e-10, maxdim = 128,
)
```

The driver [`piecewise_constant_tdvp`](@ref) also accepts an `H0`/`Ht`
pair directly, which is the more general form when the time dependence
is not channel-separable:

```julia
piecewise_constant_tdvp(Hzz, t -> ramp(t) * Hx, ψ0, 0.0, 10.0; nsteps = 200)
```

Note the sign convention: `generator_prefactor` defaults to `-im`, so
each step applies `exp(-i·dt·H)`. This differs from bare
`ITensorMPS.tdvp`, where you pass `-im*t` yourself. Pass
`generator_prefactor = -1` for imaginary-time evolution.

## Step calibration

A `schedule(t, ψ) -> TDVPStepSpec` callback chooses, per step, between
1-site and 2-site TDVP and whether to do a global Krylov subspace
expansion first (needed to grow the bond dimension under 1-site TDVP):

```julia
schedule = (t, ψ) -> maxlinkdim(ψ) < 128 ?
    TDVPStepSpec(; nsite = 2) :
    TDVPStepSpec(; nsite = 1, expand_krylov = true,
                   expand_kwargs = (; krylovdim = 2, cutoff = 1e-8))
```

!!! warning
    Never select `nsite = 1` without `expand_krylov = true` when the
    state cannot already represent the target entanglement. One-site
    TDVP cannot grow the bond dimension, so starting from a product state
    it stays at bond dimension 1 and the result is simply wrong —
    measured error 0.82 (trace distance), *independent of step count*, on
    a case where two-site TDVP reaches 1e-7. The default schedule uses
    `nsite = 2`.

Other keywords: `eval_at` (`:midpoint` or `:start`), `combine_kwargs`,
`step_observer!(; step, t_start, t_stop, state)`, `outputlevel`. An
explicit non-uniform time grid can be passed in place of
`t_start, t_stop`.

## Reference

```@docs
piecewise_constant_tdvp
TDVPStepSpec
default_tdvp_schedule
DrivenHamiltonian
```
