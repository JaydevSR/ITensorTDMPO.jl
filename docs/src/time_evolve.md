```@meta
CurrentModule = TDVPlus
```

# The `time_evolve` interface

[`time_evolve`](@ref) is the single entry point for every algorithm in
this package. The Hamiltonian is always given as [driving channels](@ref
"Driving channels"); only `alg` changes which integrator runs.

```julia
ψ = time_evolve(channels, ψ0, 0.0, 10.0; nsteps = 100)                        # magnus (default)
ψ = time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", nsteps = 100)
ψ = time_evolve(channels, ψ0, 0.0, 10.0; alg = "dyson", order = 3, nsteps = 100)
```

Each algorithm also has a direct driver — [`piecewise_constant_tdvp`](@ref),
[`dyson_evolve`](@ref), [`magnus_evolve`](@ref), [`cfet_evolve`](@ref) —
which `time_evolve` forwards to; use those when you want a method-specific
option without going through `alg_kwargs`.

## Keywords

| keyword | default | meaning |
|---|---|---|
| `alg` | `"magnus"` | integrator; a `String`, `Symbol` or `ITensors.Algorithm` |
| `nsteps` / `dt` | — | time grid, following `ITensorMPS.tdvp` conventions; or pass `times` directly |
| `order` | `2` | expansion order — Magnus 1–3, Dyson `≥ 0`. Errors for `"piecewise_constant"`, which does not expand the step |
| `cutoff` / `maxdim` | `1e-10` / `typemax(Int)` | truncation of the evolving **state** |
| `operator_cutoff` / `operator_maxdim` | `1e-12` / `typemax(Int)` | truncation of the **operators** built along the way — `Ω`, the Dyson MPO, the frozen `H(t)` |
| `generator_prefactor` | `-im` | `-im` for real time, `-1` for imaginary time |
| `normalize` | per-algorithm | renormalize after each step; needed for imaginary time |
| `adaptive` | `false` | choose step sizes automatically to meet `tol` — see [Adaptive stepping](@ref) |
| `tol` | `1e-6` | target local error per step when `adaptive = true` |
| `step_observer!` | `nothing` | called as `step_observer!(; step, t_start, t_stop, state)` after each step |
| `outputlevel` | `0` | `≥ 1` prints progress |
| `alg_kwargs` | `(;)` | forwarded verbatim to the underlying driver (`schedule`, `eval_at`, `generator_prefactor`, `normalize`, `npoints`) |

Unknown algorithm names raise an `ArgumentError` listing the valid ones
([`EVOLUTION_ALGORITHMS`](@ref)).

## Which algorithm to use

**Use `alg = "magnus"`** (the default) or, better still, `alg = "cfet"`
(see [Commutator-free propagator (CFET)](@ref), which measured faster
*and* more accurate). Avoid `alg = "dyson"` beyond small systems or short
steps — see [Scope and limitations](@ref) for why its accuracy degrades
with chain length.

Measured infidelity against a converged reference for a driven TFIM
chain at fixed `dt = 0.05`, order 2, versus chain length:

| N | `piecewise_constant_tdvp` | `dyson_evolve` | `magnus_evolve` |
|---|---|---|---|
| 4 | 9.9e-8 | 5.0e-7 | **3.9e-11** |
| 6 | 1.5e-7 | 2.8e-6 | **6.6e-11** |
| 10 | 2.5e-7 | 2.9e-5 | **1.2e-10** |
| 14 | 3.6e-7 | 1.4e-4 | **1.7e-10** |
| 20 | 5.1e-7 | 7.9e-4 | **2.5e-10** |

Over that range the frozen and Magnus errors both grow roughly linearly
in `N`, while the Dyson error grows by a factor of ~1600 (≈ `N^4.7` for
this model). `dyson_evolve` is already worse than simply freezing the
Hamiltonian by `N ≈ 4`, and the gap widens quickly — see
[Scope and limitations](@ref) for the mechanism.

## Equal-error runtime: Magnus vs. piecewise-constant TDVP

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

Magnus costs about twice as much per step but converges at roughly
fourth order rather than second, so the equal-accuracy speedup *grows*
as the tolerance tightens:

| target error | `magnus_evolve` order 2 | `piecewise_constant_tdvp` | speedup |
|---|---|---|---|
| 5.6e-4 | 8 steps, 4.4 s | ~65 steps, 16 s | **3.7×** |
| 3.7e-5 | 16 steps, 9.4 s | ~252 steps, 64 s | **6.8×** |
| 2.4e-6 | 32 steps, 19 s | ~988 steps, 249 s | **13.3×** |

Asymptotically the ratio scales as `ε^(-1/4)`: reaching error `ε` needs
`ns ∝ ε^(-1/2)` segmented steps but only `ns ∝ ε^(-1/4)` Magnus steps.
Below about `1e-6` the order-2 Magnus error flattens against the
internal TDVP exponentiation error — raise `tdvp_kwargs.nsteps` there.

`CFET` (see its own page) beats Magnus on both axes simultaneously and
is the better default in practice.

## Imaginary time

Pass `generator_prefactor = -1` (with `normalize = true`, since the
evolution is no longer unitary) to project toward the ground state:

```julia
ψ = time_evolve(channels, ψ0, 0.0, 6.0;
                alg = "cfet", nsteps = 60,
                generator_prefactor = -1, normalize = true)
```

This works for every algorithm: the factor is carried by the
time-ordered integrals for Magnus and Dyson, and applied directly to
each exponent for CFET and the piecewise-constant driver.

## Reference

```@docs
time_evolve
EVOLUTION_ALGORITHMS
```
