```@meta
CurrentModule = ITensorTDMPO
```

# Dyson series

```
U(t, t₀) = 1 + Σₐ [fₐ] H⁽ᵃ⁾ + Σₐᵦ [fₐfᵦ] H⁽ᵃ⁾H⁽ᵇ⁾ + …
```

```julia
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "dyson", order = 2, nsteps = 100)

# ...or the driver directly, for independent control of each truncation
U = dyson_mpo_fsm(channels, t0, t1; order = 2, cutoff = 1e-14)
ψ = dyson_evolve(
    channels, ψ0, 0.0, 10.0;
    nsteps = 100, order = 2,
    mpo_kwargs = (; cutoff = 1e-14),
    apply_kwargs = (; cutoff = 1e-10, maxdim = 128),
)
```

A truncated Dyson MPO is not exactly unitary, so [`dyson_evolve`](@ref)
renormalizes after each step by default (`normalize = true`).

For a single channel with constant driving the construction reduces to
the truncated Taylor series of `exp(-i Δ H)`, which the tests verify
against dense matrices.

## Two constructions

[`dyson_evolve`](@ref) builds each step with [`dyson_mpo_fsm`](@ref),
the size-extensive finite-state-machine encoding of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597) — its accuracy does
not degrade with chain length.

A second, *direct* construction is also implemented and exported —
[`dyson_mpo`](@ref) and [`dyson_terms`](@ref) — which builds operator
strings by explicit MPO multiplication rather than by manipulating the
paper's `{L, R, A, D}` block structure. It is simpler and useful for
inspecting individual terms of the series directly, and it is what the
FSM construction is verified against, but it is **not** size-extensive:
its error grows sharply with chain length, and by around 4 sites it is
already less accurate than freezing the Hamiltonian. Prefer
[`dyson_evolve`](@ref)/[`dyson_mpo_fsm`](@ref) unless you specifically
need [`dyson_mpo`](@ref)'s direct term-by-term construction. See
[Scope and limitations](@ref) for the measured comparison and why the
paper's row compression is not implemented on top of the FSM
construction.

## Why Dyson improves order-by-order but Magnus doesn't

The paper's own benchmark (Sec. VIII A) shows the Dyson MPO's error
scaling cleanly as `O(dtᴺ)` at every order `N`. Both constructions
encode the same series to the same order, so this applies equally to
[`dyson_mpo_fsm`](@ref) and [`dyson_mpo`](@ref); the table below was
measured on the latter, against an exact dense reference:

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
requires `Ω₃` *and* `Ω₄` together, which is a substantially harder
nested-commutator basis to get right. See [Magnus expansion](@ref) for
why `Ω₄` is not offered here and where higher order comes from instead.

## When Dyson wins on cost

The previous section shows Dyson needs a higher nominal order than
Magnus/CFET to reach the same *global* accuracy — but "higher order"
does not mean "more expensive" until construction cost catches up with
it, and at low order it hasn't. Measured on the driven TFIM chain
(`Sz·Sz` + oscillating transverse field) at `N = 12` with exact bond
dimension, against a dense RK4 reference, finding the cheapest
configuration of each method that reaches a given target error:

| target error | cheapest method | cost |
|---|---|---|
| `~1e-6` | `dyson` order 3, 8 steps (`1.22e-6`) | **1.05 s** |
| `~1e-8` | `dyson` order 3, 16 steps (`1.60e-8`) | **1.38 s** |
| `~1e-10` | `cfet`, 16 steps (`6.44e-12`) | 12.95 s |

At moderate accuracy — plausibly where most workloads sit — Dyson at
order 2–3 is **3–6× cheaper** than `cfet`/`magnus` for comparable or
better accuracy, because a low-order Dyson MPO is cheap to build and
[`dyson_evolve`](@ref) applies it in a single `apply` per step, with no
internal TDVP sweep at all (see [Large-system benchmarks](@ref) for the
same effect at production scale). But this reverses at tight targets:
reaching `~1e-10` needs order 4, and order 4 (16 steps, `5.62e-11`,
22.46 s) is already *slower* than `cfet`/`magnus` reaching *better*
accuracy. Past order 4 it gets much worse fast — order 5 alone (8
steps) measured `12.31 s → 610.75 s` going from order 4, a ~50×
jump for one more order, the combinatorial construction cost described
in [Scope and limitations](@ref) arriving in full force. Don't reach
for `order ≥ 5` expecting it to behave like the cheap end of this
table.

**So: `cfet` remains the right default** — it is uniformly good across
the whole accuracy range, with no such cliff — **but if you know your
accuracy target is moderate (order 2–3 territory) and want the fastest
option, `dyson_evolve` at that order is a genuine, measured win, not a
compromise.** This is a distinct comparison from the previous section:
that one holds order fixed and asks which method is more accurate;
this one holds the *target accuracy* fixed and asks which method is
cheaper to reach it — and the answer depends on where in the accuracy
range you're asking.

## Reference

```@docs
dyson_evolve
dyson_mpo_fsm
dyson_mpo
dyson_terms
```
