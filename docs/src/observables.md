```@meta
CurrentModule = TDVPlus
```

# Observables and diagnostics

## Observables

[`EvolutionObserver`](@ref) collects measurements along the evolution
and is itself the `step_observer!` callback:

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

Each measurement is called as `f(state)` or, if it accepts two
arguments, `f(state, t)`. Element types are narrowed, so `r.entropy` is
a `Vector{Float64}`. Pass `every = k` to record only every `k`-th step —
use this for diagnostics that cost more than the evolution itself.

## Adiabaticity diagnostics

Everything here is **opt-in**: no driver computes a diagnostic unless
you put it in an observer.

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

## Reference

```@docs
EvolutionObserver
observe!
results
entanglement_entropy
entanglement_profile
instantaneous_energy
energy_variance
instantaneous_spectrum
instantaneous_gap
adiabatic_report
```
