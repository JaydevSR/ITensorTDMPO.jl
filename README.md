# ITensorMPSExtended.jl

Personal extensions to [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl)
for research use. The current focus is **time evolution under time-dependent
Hamiltonians**, in particular adiabatic ramps.

## Installation

Not registered. From the Julia REPL:

```julia
using Pkg
Pkg.develop(path="path/to/ITensorMPSExtended.jl")
```

## Overview

The Hamiltonian is always described the same way — as a set of **driving
channels** `H(t) = Σₐ fₐ(t) H^{(a)}`, each a time-independent MPO paired
with a scalar driving function — and a single entry point selects the
integrator:

```julia
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                nsteps = 100, cutoff = 1e-10, maxdim = 128)

# Same Hamiltonian, different integrator — nothing else changes.
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "piecewise_constant", nsteps = 100)
```

Channels are given as a list of `(driving, MPO)` tuples. The MPO and the
driving are told apart by type, so `(H, f)` works as well as `(f, H)`, and
`H => f` pairs are accepted too.

A driving is either **a function of time** or **a plain number** for a
constant coefficient — `(J, H)` is shorthand for the static term `J * H`:

```julia
[(1.0, Hzz), (ramp, Hx)]     # unit coupling
[(-2.5, Hzz), (ramp, Hx)]    # any constant
[(one, Hzz), (ramp, Hx)]     # a function works too; `one` ≡ 1.0
```

A driving that is neither callable nor a number is rejected when the
channels are built, rather than failing later inside the quadrature.

If you want to reuse the decomposition across calls, build it explicitly
with `DrivingChannels([(1.0, Hzz), (ramp, Hx)])` and pass that instead —
every entry point accepts either.

Three algorithms are available, in increasing order of sophistication:

| `alg` | Treatment of `H(t)` within a step | Order | Unitary? |
|---|---|---|---|
| `"piecewise_constant"` | frozen at one evaluation point | 2 | yes (TDVP) |
| `"dyson"` | expanded to order `N` in the Dyson series | `N` | approximately |
| `"magnus"` *(default)* | `Ω₁ + Ω₂` (+`Ω₃`), applies `exp(Ω)` | ~4 | yes |
| `"cfet"` | product of exponentials at Gauss nodes — **no commutators** | 4 | yes |

**`"cfet"` measured strictly better than `"magnus"`** — roughly 2× faster
*and* ~1.5× more accurate at equal step count (see
[Commutator-free propagator](#commutator-free-propagator-cfet)). It is not
the default only because `"magnus"` was there first; switching is a
one-word change.

Each also has a direct driver — [`piecewise_constant_tdvp`](#piecewise-constant-tdvp),
`dyson_evolve`, `magnus_evolve` — which `time_evolve` forwards to; use those
when you want a method-specific option without going through `alg_kwargs`.

### Keywords

| keyword | default | meaning |
|---|---|---|
| `alg` | `"magnus"` | integrator; a `String`, `Symbol` or `ITensors.Algorithm` |
| `nsteps` / `dt` | — | time grid, following `ITensorMPS.tdvp` conventions; or pass `times` directly |
| `order` | `2` | expansion order — Magnus 1–3, Dyson `≥ 0`. Errors for `"piecewise_constant"`, which does not expand the step |
| `cutoff` / `maxdim` | `1e-10` / `typemax(Int)` | truncation of the evolving **state** |
| `operator_cutoff` / `operator_maxdim` | `1e-12` / `typemax(Int)` | truncation of the **operators** built along the way — `Ω`, the Dyson MPO, the frozen `H(t)` |
| `generator_prefactor` | `-im` | `-im` for real time, `-1` for imaginary time |
| `normalize` | per-algorithm | renormalize after each step; needed for imaginary time |
| `adaptive` | `false` | choose step sizes automatically to meet `tol` |
| `tol` | `1e-6` | target local error per step when `adaptive = true` |
| `step_observer!` | `nothing` | called as `step_observer!(; step, t_start, t_stop, state)` after each step |
| `outputlevel` | `0` | `≥ 1` prints progress |
| `alg_kwargs` | `(;)` | forwarded verbatim to the underlying driver (`schedule`, `eval_at`, `generator_prefactor`, `normalize`, `npoints`) |

Unknown algorithm names raise an `ArgumentError` listing the valid ones
(`EVOLUTION_ALGORITHMS`).

The Dyson and Magnus drivers implement the constructions of
[Vanthilt, Van Damme, Haegeman, McCulloch & Vanderstraeten,
*Matrix Product Operator Encodings of the Magnus Expansion and Dyson
Series*](https://arxiv.org/abs/2605.21597) — see [Scope](#scope) for what
is and is not implemented.

### Which driver to use

**Use `alg = "magnus"`** (the default). Measured infidelity against a
converged reference for a driven TFIM chain at fixed `dt = 0.05`, order 2,
versus chain length:

| N | `piecewise_constant_tdvp` | `dyson_evolve` | `magnus_evolve` |
|---|---|---|---|
| 4 | 9.9e-8 | 5.0e-7 | **3.9e-11** |
| 6 | 1.5e-7 | 2.8e-6 | **6.6e-11** |
| 10 | 2.5e-7 | 2.9e-5 | **1.2e-10** |
| 14 | 3.6e-7 | 1.4e-4 | **1.7e-10** |
| 20 | 5.1e-7 | 7.9e-4 | **2.5e-10** |

Over that range the frozen and Magnus errors both grow roughly linearly in
`N`, while the Dyson error grows by a factor of ~1600 (≈ `N^4.7` for this
model). **`dyson_evolve` is already worse than simply freezing the
Hamiltonian by N ≈ 4, and the gap widens quickly.**

The reason is the size-extensivity issue at the heart of the paper.
`magnus_evolve` builds the generator `Ω` and applies `exp(Ω)`, and the
exponential resums all *disjoint* higher-order products correctly to all
orders — so extensivity comes for free. `dyson_evolve` truncates a
polynomial in `H`, discarding an order-`N+1` term whose disjoint part
scales as `N^{N+1}`, with nothing to resum it. The paper's size-extensive
MPO encoding is exactly what repairs this, and it is the part not
implemented here (see [Scope](#scope)).

So `dyson_mpo` is best regarded as a reference implementation of the
series — useful for small systems, short steps, and for checking the
Magnus results against — rather than a production driver.

### Magnus vs. piecewise-constant TDVP: equal-error runtime

Measured on the paper's modulated TFIM over one period at `N = 8` with
exact bond dimension (so truncation contributes nothing), against an
exact dense RK4 reference:

| method | convergence order | cost per step |
|---|---|---|
| `piecewise_constant_tdvp` | **2.0** | 1.0× |
| `magnus_evolve` order 2 | **~3.9** | ~2.1× |
| `magnus_evolve` order 3 | ~3.8 | ~2.6× |

To be precise about where these orders come from: TDVP is *not* the
limiting factor in either. Two-site TDVP applied to a time-independent
generator at exact bond dimension is accurate to ~1e-7 within a few
sweeps — it acts as an essentially exact exponentiator. The second order
of `piecewise_constant_tdvp` is entirely the *Hamiltonian-freezing*
error: switching `eval_at` from `:midpoint` to `:start` drops the method
to first order (measured ratios 1.99 vs 4.00), with everything else
unchanged. So the comparison above is really midpoint freezing (2nd
order) versus `Ω₁ + Ω₂` with exact time-ordered integrals (4th order),
both exponentiated by the same near-exact TDVP.

With truncation, TDVP additionally contributes a projection error, but
that is controlled by bond dimension rather than by `dt` and so does not
change these orders.

Magnus costs about twice as much per step but converges at roughly fourth
order rather than second, so the equal-accuracy speedup *grows* as the
tolerance tightens:

| target error | `magnus_evolve` order 2 | `piecewise_constant_tdvp` | speedup |
|---|---|---|---|
| 5.6e-4 | 8 steps, 4.4 s | ~65 steps, 16 s | **3.7×** |
| 3.7e-5 | 16 steps, 9.4 s | ~252 steps, 64 s | **6.8×** |
| 2.4e-6 | 32 steps, 19 s | ~988 steps, 249 s | **13.3×** |

Asymptotically the ratio scales as `ε^(-1/4)`: reaching error `ε` needs
`ns ∝ ε^(-1/2)` segmented steps but only `ns ∝ ε^(-1/4)` Magnus steps.
Below about `1e-6` the order-2 Magnus error flattens against the internal
TDVP exponentiation error — raise `tdvp_kwargs.nsteps` there.

Note that order 3 was not better than order 2 in this test: both converge
at fourth order, and order 3 costs more per step. `Ω₁ + Ω₂` with exact
time-ordered integrals is already a fourth-order integrator, so **order 2
is the sensible default**.

Reaching sixth order would require `Ω₄`. An implementation of it measured
~3.7th order rather than the expected 6th — the nested-commutator basis
was wrong or incomplete — and was **removed rather than shipped
unverified**. `Ω₃` is retained but is not recommended. For higher order,
[`"cfet"`](#commutator-free-propagator-cfet) is both the cheaper and the
more extensible route, since commutator-free schemes are specified by
coefficient tables rather than commutator algebra.

---

## Commutator-free propagator (CFET)

`alg = "cfet"` writes each step as a product of exponentials of *plain
weighted sums* of the channel operators, evaluated at Gauss–Legendre
nodes:

```
U(t+h, t) ≈ exp(-i h W₂) exp(-i h W₁),   Wⱼ = Σₐ wⱼₐ H⁽ᵃ⁾
```

It reaches fourth order **without ever forming a commutator**. Since
commutators are what inflate MPO bond dimension in the Magnus route, each
generator here is no larger than `H(t)` itself — the cost is two TDVP
applications per step instead of one, which turns out to be a bargain:

| steps | `"magnus"` order 2 | `"cfet"` order 4 |
|---|---|---|
| 8 | 7.45 s, err 5.6e-4 | **4.13 s, err 3.8e-4** |
| 16 | 14.95 s, err 3.7e-5 | **7.16 s, err 2.4e-5** |
| 32 | 29.00 s, err 2.5e-6 | **12.44 s, err 1.7e-6** |

Measured convergence order 3.97 / 3.99 / 3.86 against an exact dense
reference. `order = 2` degenerates to the exponential midpoint rule and
reproduces `"piecewise_constant"` exactly (measured order 2.00), which is
a useful cross-check that the two share a limit.

!!! warning "A floor, not unbounded convergence"
    Refining the step count only helps up to a point. Each step still
    runs through TDVP, and — as documented under
    [Which driver to use](#which-driver-to-use) — a fixed generator
    exponentiated by 2-site TDVP has its own roundoff floor: accuracy
    improves for the first few sweeps, then *degrades* as more, smaller
    sweeps accumulate per-application roundoff. For the benchmark model
    here that floor sits around `nsteps ≈ 16`; refining well past it can
    make CFET (or Magnus) *less* accurate, not more. Consequently, do not
    validate a fine-step run against the same method at an even finer
    step count as "ground truth" — it can itself be sitting on that floor,
    which silently invalidates the comparison. The test suite learned
    this the hard way and now checks against an independent dense RK4
    reference instead.

```julia
ψ = time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", nsteps = 100)
```

## Adaptive stepping

Set an accuracy instead of a step count:

```julia
ψ = time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", adaptive = true, tol = 1e-7)

# adaptive_time_evolve also returns the step history
ψ, hist = adaptive_time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", tol = 1e-7)
hist.dts, hist.errors      # where the stepper had to slow down
```

Each candidate step is taken once at `dt` and again as two steps of
`dt/2`; the [`trace_distance`](@ref) between the results estimates the
local error, the step is accepted when that is below `tol`, and the next
step size is scaled by `(tol/err)^(1/p)`. This costs three sub-steps per
accepted step, so it pays off when `‖Ḣ‖` varies strongly across the
evolution — a ramp that must crawl through a gap minimum and can sprint
elsewhere — and not for a uniform drive.

!!! warning "`tol` has a floor too"
    Step-doubling assumes the per-step integrator is otherwise exact, so
    that shrinking the step always shrinks the error. TDVP is not exact:
    push `tol` far enough below its own per-application roundoff and the
    coarse/fine comparison stops measuring the integration error at all —
    it measures roundoff noise instead. Measured on the benchmark model
    above, the true error (against an independent reference) is minimized
    around `tol ≈ 1e-7` and *increases* for `tol = 1e-8, 1e-9, 1e-10`,
    even though the stepper dutifully takes more steps each time:

    | `tol` | true error | steps |
    |---|---|---|
    | 1e-5 | 7.4e-7 | 6 |
    | 1e-6 | 3.8e-7 | 8 |
    | **1e-7** | **3.4e-7 (best)** | 12 |
    | 1e-8 | 8.7e-7 | 21 |
    | 1e-10 | 1.3e-6 | 34 |

    A telltale sign you've crossed this floor: the recorded step errors in
    `hist.errors` start reading exactly `0.0`. Keep `tol` at or above
    roughly the state/operator `cutoff` in use, not many orders tighter.

## Imaginary time

Pass `generator_prefactor = -1` (with `normalize = true`, since the
evolution is no longer unitary) to project toward the ground state:

```julia
ψ = time_evolve(channels, ψ0, 0.0, 6.0;
                alg = "cfet", nsteps = 60,
                generator_prefactor = -1, normalize = true)
```

This works for every algorithm: the factor is carried by the time-ordered
integrals for Magnus and Dyson, and applied directly to each exponent for
CFET and the piecewise-constant driver.

## Observables

`EvolutionObserver` collects measurements along the evolution and is
itself the `step_observer!` callback:

```julia
obs = EvolutionObserver(
    :chi      => maxlinkdim,
    :entropy  => ψ -> entanglement_entropy(ψ),
    :energy   => (ψ, t) -> instantaneous_energy(channels, t, ψ),
)

observe!(obs, ψ0, 0.0)                     # optional: record the initial state
ψ = time_evolve(channels, ψ0, 0.0, 10.0; nsteps = 100, step_observer! = obs)

r = results(obs)                            # (; step, time, chi, entropy, energy)
```

Each measurement is called as `f(state)` or, if it accepts two arguments,
`f(state, t)`. Element types are narrowed, so `r.entropy` is a
`Vector{Float64}`. Pass `every = k` to record only every `k`-th step —
use this for diagnostics that cost more than the evolution itself.

## Adiabaticity diagnostics

Everything here is **opt-in**: no driver computes a diagnostic unless you
put it in an observer.

| function | cost | what it tells you |
|---|---|---|
| `instantaneous_energy(ch, t, ψ)` | one MPO application | `⟨H(t)⟩` |
| `energy_variance(ch, t, ψ)` | one MPO application | `⟨H²⟩ − ⟨H⟩²`, zero on an eigenstate |
| `entanglement_entropy(ψ, b)` | one SVD | von Neumann entropy across bond `b` |
| `instantaneous_gap(ch, t, ψ)` | **DMRG solve** | `E₁(t) − E₀(t)` |
| `adiabatic_report(ch, t, ψ)` | cheap by default | bundle; `gap = true` adds DMRG |

**Energy variance is the cheap adiabaticity check.** It vanishes exactly
when the state is an instantaneous eigenstate, needs no ground-state
solve, and never forms `H²` (it is computed from `H|ψ⟩`). Use it every
step; reserve the DMRG-based gap for a sparse subset via `every = k`.

```julia
adiabatic_report(ch, t, ψ)              # energy + variance, no DMRG
adiabatic_report(ch, t, ψ; gap = true)  # adds gap, excess energy, GS fidelity
```

---

## Ramps

A `Ramp` separates the **shape** of a sweep from its **endpoints**. A
`RampShape` is a pure profile on the unit interval — `s: [0,1] → [0,1]`
with `s(0) = 0` and `s(1) = 1` — and `Ramp` maps that profile onto
physical time and physical values:

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

Clamping happens inside the shape, so outside `[t_start, t_stop]` the ramp
*holds* at its endpoint value rather than extrapolating — you can keep
evolving past `t_stop` and the parameter stays put.

```julia
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)
ramp(0.0)   # 0.0
ramp(5.0)   # 1.0   (midpoint of a symmetric shape)
ramp(10.0)  # 2.0
ramp(50.0)  # 2.0   (held, not extrapolated)
```

Ramps run downward as readily as upward (`value_start > value_stop`), and
nothing requires a `Ramp` at all — any `f(t)` is a valid driving, as is a
plain number for a constant.

### Shapes

| Shape | `s(τ)` | endpoint slope |
|---|---|---|
| `LinearRamp()` | `τ` | **nonzero** at both ends |
| `PowerLawRamp(p)` | `τᵖ` | zero at start for `p > 1` |
| `ExponentialRamp(k)` | `(e^{kτ} − 1)/(e^k − 1)` | asymmetric; `k = 0` is linear |
| `SineRamp()` | `(1 − cos πτ)/2` | zero at both ends |
| `SineSquaredRamp()` | `sin²(πτ/2)` | identical to `SineRamp` (half-angle identity) |
| `SmoothstepRamp()` | `3τ² − 2τ³` | zero at both ends |
| `SmootherstepRamp()` | `6τ⁵ − 15τ⁴ + 10τ³` | zero slope **and** curvature at both ends |

For adiabatic sweeps this choice is substantive. `LinearRamp` switches the
sweep on discontinuously in `Ḣ`, and that kink is a broadband perturbation
which drives diabatic transitions no matter how slowly you ramp. Shapes
with `s′(0) = s′(1) = 0` turn the drive on and off smoothly and suppress
those endpoint excitations; `SmootherstepRamp` additionally kills the
curvature. `PowerLawRamp` and `ExponentialRamp` are the asymmetric
options — useful when you want to move slowly through a gap minimum at one
end only.

## Piecewise-constant TDVP

The simplest algorithm: freeze `H(t)` once per interval and run an ordinary
TDVP sweep across it.

```julia
using ITensorMPS, ITensorMPSExtended

sites = siteinds("S=1/2", 20)
Hzz = MPO(OpSum() + ..., sites)          # coupling
Hx  = MPO(OpSum() + ..., sites)          # transverse field
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

ψ = time_evolve(
    [(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
    alg = "piecewise_constant", nsteps = 200, cutoff = 1e-10, maxdim = 128,
)
```

The driver `piecewise_constant_tdvp` also accepts an `H0`/`Ht` pair
directly, which is the more general form when the time dependence is not
channel-separable:

```julia
piecewise_constant_tdvp(Hzz, t -> ramp(t) * Hx, ψ0, 0.0, 10.0; nsteps = 200)
```

Note the sign convention: `generator_prefactor` defaults to `-im`, so each
step applies `exp(-i·dt·H)`. This differs from bare `ITensorMPS.tdvp`, where
you pass `-im*t` yourself. Pass `generator_prefactor = -1` for imaginary-time
evolution.

**Step calibration.** A `schedule(t, ψ) -> TDVPStepSpec` callback chooses,
per step, between 1-site and 2-site TDVP and whether to do a global Krylov
subspace expansion first (needed to grow the bond dimension under 1-site
TDVP):

```julia
schedule = (t, ψ) -> maxlinkdim(ψ) < 128 ?
    TDVPStepSpec(; nsite = 2) :
    TDVPStepSpec(; nsite = 1, expand_krylov = true,
                   expand_kwargs = (; krylovdim = 2, cutoff = 1e-8))
```

!!! warning
    Never select `nsite = 1` without `expand_krylov = true` when the state
    cannot already represent the target entanglement. One-site TDVP cannot
    grow the bond dimension, so starting from a product state it stays at
    bond dimension 1 and the result is simply wrong — measured error 0.82
    (trace distance), *independent of step count*, on a case where two-site
    TDVP reaches 1e-7. The default schedule uses `nsite = 2`.

Other keywords: `eval_at` (`:midpoint` or `:start`), `combine_kwargs`,
`step_observer!(; step, t_start, t_stop, state)`, `outputlevel`. An explicit
non-uniform time grid can be passed in place of `t_start, t_stop`.

---

## Driving channels

The Dyson and Magnus constructions both start from the paper's
decomposition of the Hamiltonian into **driving channels**,

```
H(t) = Σₐ fₐ(t) H⁽ᵃ⁾
```

where each `H⁽ᵃ⁾` is a *time-independent* MPO and each `fₐ` a scalar driving
function. This is what makes the expansions tractable: the operator content
(products and commutators of the `H⁽ᵃ⁾`) is computed once, and all time
dependence enters through scalar integrals of the `fₐ`.

```julia
channels = DrivingChannels([(1.0, Hzz), (ramp, Hx)])
channels(0.5)        # assembles H(0.5) as an MPO (useful for testing)
nchannels(channels)  # 2
```

`time_evolve` builds this internally when handed a plain list, so you only
need it explicitly to reuse one decomposition across several calls.

## Time-ordered integrals

The scalar prefactors in both expansions are the time-ordered integrals

```
[f₁ f₂ … fₙ] = (-i)ⁿ ∫dt₁ ∫^{t₁}dt₂ … ∫^{t_{n-1}}dtₙ  f₁(t₁) f₂(t₂) … fₙ(tₙ)
```

with `t₁ > t₂ > … > tₙ`, so `f₁` is evaluated at the *latest* time. The
factors of `-i` are folded into the bracket.

```julia
time_ordered_integral([f, g], t0, t1)     # [f g]
time_ordered_integral(fill(one, 3), 0, Δ) # ≈ (-i)³ Δ³/3!
```

These are evaluated by repeated cumulative Simpson quadrature on a uniform
grid — `O(n · npoints)` rather than the `O(npointsⁿ)` of naive nested
quadrature. The paper instead uses a quantics tensor-train representation
([QuanticsTT.jl](https://github.com/VictorVanthilt/QuanticsTT.jl)), which
matters at high order; the grid approach here is simpler and adequate for
the orders implemented.

The implementation satisfies the paper's factoring property
`[fₐfᵦ] + [fᵦfₐ] = [fₐ][fᵦ]`, which is checked in the test suite.

## Dyson series

```
U(t, t₀) = 1 + Σₐ [fₐ] H⁽ᵃ⁾ + Σₐᵦ [fₐfᵦ] H⁽ᵃ⁾H⁽ᵇ⁾ + …
```

```julia
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "dyson", order = 2, nsteps = 100)

# ...or the driver directly, for independent control of each truncation
U = dyson_mpo(channels, t0, t1; order = 2, cutoff = 1e-12)
ψ = dyson_evolve(
    channels, ψ0, 0.0, 10.0;
    nsteps = 100, order = 2,
    mpo_kwargs = (; cutoff = 1e-12),
    apply_kwargs = (; cutoff = 1e-10, maxdim = 128),
)
```

A truncated Dyson MPO is not exactly unitary, so `dyson_evolve` renormalizes
after each step by default (`normalize = true`).

For a single channel with constant driving the construction reduces to the
truncated Taylor series of `exp(-i Δ H)`, which the tests verify against
dense matrices.

## Magnus expansion

`U(t,t₀) = exp(Ω(t,t₀))` with

```
Ω₁ = Σₐ [fₐ] H⁽ᵃ⁾
Ω₂ = ½ Σₐᵦ [fₐfᵦ] [H⁽ᵃ⁾, H⁽ᵇ⁾]
Ω₃ = ⅙ Σₐᵦ𝚌 [fₐfᵦf𝚌] ( [[H⁽ᵃ⁾,H⁽ᵇ⁾],H⁽ᶜ⁾] + [H⁽ᵃ⁾,[H⁽ᵇ⁾,H⁽ᶜ⁾]] )
```

Orders 1–3 are supported; **order 2 is the right choice** (order 3
converges no faster and costs more). See the note above on why `Ω₄` is
absent.

```julia
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                order = 2, nsteps = 100)     # "magnus" is the default

# ...or the driver directly
Ω = magnus_generator(channels, t0, t1; order = 2)
ψ = magnus_evolve(
    channels, ψ0, 0.0, 10.0;
    nsteps = 100, order = 2,
    tdvp_kwargs = (; cutoff = 1e-10, maxdim = 128),
)
```

Because the brackets carry the factors of `-i`, `Ω` is anti-Hermitian and
`exp(Ω)` is unitary — so unlike the Dyson driver, `magnus_evolve` preserves
the norm exactly. The exponential is applied as `tdvp(Ω, 1.0, ψ)`: the
generator already carries the whole step, so the TDVP "time" is 1, matching
the paper's exponentiation of `Ω` at `τ = 1`.

The commutator structure is what keeps each term extensive — the
non-overlapping ("disjoint") contributions cancel between `AB` and `BA`.

---

## Scope

**What is implemented:** the paper's *physical structure* — the driving-channel
decomposition, the time-ordered integrals, and the Dyson and Magnus expansions
expressed in that language, to arbitrary order (Dyson) and to third order
(Magnus).

**What is not:** the paper's central technical contribution, the
*size-extensive finite-state-machine MPO encoding*. Operator strings and
commutators here are formed by direct MPO multiplication
(`apply(A, B)`) rather than by manipulating the `{L, R, A, D}` block
structure of first-degree MPOs at the level of virtual "levels", and neither
the exact equivalent-column compression nor the approximate row compression
of the paper's Sec. VII is implemented.

The practical consequences:

- **Finite systems only.** `ITensorMPS` represents only finite MPS/MPO, so
  everything here is finite by construction; infinite systems would need
  `ITensorInfiniteMPS.jl` or MPSKit.jl.
- **Accuracy degrades with chain length for `dyson_evolve`**, because the
  direct construction is not size-extensive — see the measured scaling in
  [Which driver to use](#which-driver-to-use). `magnus_evolve` is not
  affected, since exponentiating the generator resums the disjoint terms.
- **Bond dimension grows faster** than the paper's compressed construction.
  Intermediate products are truncated with `cutoff`/`maxdim` to keep this in
  check, but expect the MPO bond dimension to be the practical limit on
  order and system size.
- **Cost is exponential in the order.** The order-`N` Dyson MPO enumerates
  `nchannels^N` operator strings. Fine for `order ≤ 3` and a few channels.

Note that size-extensivity is **not** an infinite-system concern only. The
paper applies its encoding to finite systems too (Sec. VIII A benchmarks a
finite L = 8 chain at exact bond dimension and recovers clean `O(dtᴺ)`
convergence). On an infinite system non-extensivity is *fatal* — applying
`Hⁿ` to a normalized uMPS gives a state that cannot be normalized. On a
finite system it is merely *harmful*: everything is well defined, but the
accuracy degrades with chain length, which is exactly the `N^4.7` growth
measured above.

ITensors builds MPOs from `OpSum` with automatic compression and does not
expose the Jordan-block/first-degree structure that the paper's algorithms
operate on, so implementing the full encoding would mean building that layer
from scratch. That is the natural next step, and it would fix the Dyson
driver's chain-length scaling on finite systems — it is worth doing even if
you never want the thermodynamic limit.

## Testing

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The expansions are checked against **independent dense-matrix**
reference evolution on small chains, rather than against each other: the
reduction to the Taylor series for constant driving, anti-Hermiticity of
`Ω`, the equivalence of first-order Dyson and first-order Magnus noted in
the paper, the expected convergence order under step refinement, and that
both expansions beat freezing the Hamiltonian for a rapidly oscillating
drive. Also covered: QN-conserving (`conserve_qns = true`) evolution with
flux preservation, the time-ordered-integral closed forms and the
factoring property, that every `time_evolve` algorithm reproduces its
direct driver exactly, the CFET weights and its fourth-order convergence,
adaptive stepping against its own tolerance, imaginary time converging to
the DMRG ground-state energy, and the observable/diagnostic layer
(entropy of a Bell pair, zero variance on an eigenstate, gap against a
direct DMRG solve).

Note that these drivers do not renormalize, so the state norm drifts by
~1e-11 under truncation. Compare states with a normalized overlap
`abs(inner(a, b)) / (norm(a) * norm(b))`, not with `abs(inner(a, b))`
against 1.
