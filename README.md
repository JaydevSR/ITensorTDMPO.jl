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

Three time-evolution drivers are provided, in increasing order of
sophistication:

| Driver | Treatment of `H(t)` within a step | Unitary? |
|---|---|---|
| `piecewise_constant_tdvp` | frozen at one evaluation point | yes (TDVP) |
| `dyson_evolve` | expanded to order `N` in the Dyson series | approximately |
| `magnus_evolve` | expanded to order `N` in the Magnus expansion | yes |

The Dyson and Magnus drivers implement the constructions of
[Vanthilt, Van Damme, Haegeman, McCulloch & Vanderstraeten,
*Matrix Product Operator Encodings of the Magnus Expansion and Dyson
Series*](https://arxiv.org/abs/2605.21597) — see [Scope](#scope) for what
is and is not implemented.

### Which driver to use

**Use `magnus_evolve`.** Measured infidelity against a converged reference
for a driven TFIM chain at fixed `dt = 0.05`, order 2, versus chain length:

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

---

## Ramps

Ramp shapes map a normalized time `τ ∈ [0,1]` to a normalized progress
`s ∈ [0,1]`, with `s(0) = 0`, `s(1) = 1`, clamped outside the window.

| Shape | `s(τ)` |
|---|---|
| `LinearRamp()` | `τ` |
| `SmoothstepRamp()` | `3τ² − 2τ³` (zero slope at ends) |
| `SmootherstepRamp()` | `6τ⁵ − 15τ⁴ + 10τ³` (zero slope and curvature at ends) |
| `SineRamp()` | `(1 − cos πτ)/2` |
| `SineSquaredRamp()` | `sin²(πτ/2)` — identical to `SineRamp` by the half-angle identity |
| `PowerLawRamp(p)` | `τᵖ` |
| `ExponentialRamp(k)` | `(e^{kτ} − 1)/(e^k − 1)`; `k = 0` reduces to linear |

`Ramp(shape, t_start, t_stop, value_start, value_stop)` turns a shape into a
function of physical time:

```julia
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)
ramp(0.0)   # 0.0
ramp(5.0)   # 1.0
ramp(10.0)  # 2.0
ramp(50.0)  # 2.0  (clamped)
```

## Piecewise-constant TDVP

The simplest driver: freeze `H(t)` once per interval and run an ordinary
TDVP sweep across it.

```julia
using ITensorMPS, ITensorMPSExtended

sites = siteinds("S=1/2", 20)
Hzz = MPO(OpSum() + ..., sites)          # static part
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)
Ht = t -> ramp(t) * Hx                    # time-dependent part

ψ = piecewise_constant_tdvp(
    Hzz, Ht, ψ0, 0.0, 10.0;
    nsteps = 200,
    tdvp_kwargs = (; cutoff = 1e-10, maxdim = 128),
)
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
channels = DrivingChannels(Hzz => (t -> 1.0), Hx => ramp)
channels(0.5)        # assembles H(0.5) as an MPO (useful for testing)
nchannels(channels)  # 2
```

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

Orders 1–3 are supported.

```julia
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

ITensors builds MPOs from `OpSum` with automatic compression and does not
expose the Jordan-block/first-degree structure that the paper's algorithms
operate on, so implementing the full encoding would mean building that layer
from scratch. That is the natural next step if the bond dimension or the
thermodynamic limit becomes the binding constraint.

## Testing

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The suite checks the expansions against independent dense-matrix reference
evolution on small chains: the reduction to the Taylor series for constant
driving, anti-Hermiticity of `Ω`, the equivalence of first-order Dyson and
first-order Magnus noted in the paper, the expected convergence order under
step refinement, and that both expansions beat freezing the Hamiltonian for
a rapidly oscillating drive.
