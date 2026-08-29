```@meta
CurrentModule = ITensorTDMPO
```

# Commutator-free propagator (CFET)

The scheme — two exponentials at two-point Gauss–Legendre nodes,
weighted `(3 ± 2√3)/12` — is the standard fourth-order commutator-free
exponential time propagator for driven quantum systems, introduced by
[Alvermann & Fehske (2011)](https://doi.org/10.1016/j.jcp.2011.04.006),
*High-order commutator-free exponential time-propagation of driven
quantum systems*, J. Comput. Phys. 230, 5930.

`alg = "cfet"` writes each step as a product of exponentials of *plain
weighted sums* of the channel operators, evaluated at Gauss–Legendre
nodes:

```
U(t+h, t) ≈ exp(-i h W₂) exp(-i h W₁),   Wⱼ = Σₐ wⱼₐ H⁽ᵃ⁾
```

It reaches fourth order **without ever forming a commutator**. Since
commutators are what inflate MPO bond dimension in the Magnus route,
each generator here is no larger than `H(t)` itself — the cost is two
TDVP applications per step instead of one, which turns out to be a
bargain:

| steps | `"magnus"` order 2 | `"cfet"` order 4 |
|---|---|---|
| 8 | 7.45 s, err 5.6e-4 | **4.13 s, err 3.8e-4** |
| 16 | 14.95 s, err 3.7e-5 | **7.16 s, err 2.4e-5** |
| 32 | 29.00 s, err 2.5e-6 | **12.44 s, err 1.7e-6** |

Measured convergence order 3.97 / 3.99 / 3.86 against an exact dense
reference. `order = 2` degenerates to the exponential midpoint rule and
reproduces `"piecewise_constant"` exactly (measured order 2.00), which
is a useful cross-check that the two share a limit.

**`"cfet"` measured strictly better than `"magnus"`** — roughly 2×
faster *and* ~1.5× more accurate at equal step count — which is why it
is the default algorithm for [`time_evolve`](@ref).

```julia
ψ = time_evolve(channels, ψ0, 0.0, 10.0; nsteps = 100)   # "cfet" by default
```

!!! warning "A floor, not unbounded convergence"
    Refining the step count only helps up to a point. Each step still
    runs through TDVP, and — as documented on
    [The `time_evolve` interface](@ref) page — a fixed generator
    exponentiated by 2-site TDVP has its own roundoff floor: accuracy
    improves for the first few sweeps, then *degrades* as more, smaller
    sweeps accumulate per-application roundoff. For the benchmark model
    here that floor sits around `nsteps ≈ 16`; refining well past it can
    make CFET (or Magnus) *less* accurate, not more. Consequently, do
    not validate a fine-step run against the same method at an even
    finer step count as "ground truth" — it can itself be sitting on
    that floor, which silently invalidates the comparison. The test
    suite checks against an independent dense RK4 reference for
    exactly this reason.

## Reference

```@docs
cfet_evolve
cfet_exponents
CF4_NODES
CF4_WEIGHTS
```
