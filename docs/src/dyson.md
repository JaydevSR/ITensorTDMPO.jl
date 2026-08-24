```@meta
CurrentModule = TDVPlus
```

# Dyson series

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

A truncated Dyson MPO is not exactly unitary, so [`dyson_evolve`](@ref)
renormalizes after each step by default (`normalize = true`).

For a single channel with constant driving the construction reduces to
the truncated Taylor series of `exp(-i Δ H)`, which the tests verify
against dense matrices.

!!! warning "Prefer Magnus or CFET"
    `dyson_evolve`'s accuracy degrades with chain length — see
    [Scope and limitations](@ref) — and it should be treated as a
    reference implementation of the series (useful for small systems,
    short steps, and cross-checking the other algorithms) rather than a
    production driver.

## Why Dyson improves order-by-order but Magnus doesn't

The paper's own benchmark (Sec. VIII A) shows the Dyson MPO's error
scaling cleanly as `O(dtᴺ)` at every order `N`. Measured here, against an
exact dense reference, `dyson_mpo` reproduces that exactly:

| order | 1 | 2 | 3 |
|---|---|---|---|
| Dyson: global order | **~1.0** | **~2.0** | **~2.9** |
| Magnus: global order | **~2.0** | **~3.9** | **~3.9** |

Dyson is a **plain truncated polynomial** in the time step — `U ≈ 1 +
Σₐ[fₐ]H⁽ᵃ⁾ + Σₐᵦ[fₐfᵦ]H⁽ᵃ⁾H⁽ᵇ⁾ + …` — so order `N` means "correct
through `(dt)ᴺ`" and each additional order buys exactly one more power
of `dt`, with no further subtlety.

Magnus is different in kind: it builds a generator from nested
commutators and then *exponentiates* it, `U ≈ exp(Ω)`. That nonlinear
map is where the pairing comes from — a standard feature of
Magnus/geometric integrators, not a defect: `Ω₁` alone already reaches
global order 2 (a "free" jump from the exponential), and `Ω₁+Ω₂` jumps
again to ~4, but `Ω₃` alone doesn't buy a further jump to 6 — that
requires `Ω₃` *and* `Ω₄` together, which is exactly why an `Ω₄`
implementation (targeting 6, measuring ~3.7) failed rather than
complementing `Ω₃` and was removed rather than shipped unverified. See
[Magnus expansion](@ref) for the full story.

## Reference

```@docs
dyson_mpo
dyson_terms
dyson_evolve
```
