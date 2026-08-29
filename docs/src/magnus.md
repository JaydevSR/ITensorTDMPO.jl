```@meta
CurrentModule = ITensorTDMPO
```

# Magnus expansion

`U(t,t₀) = exp(Ω(t,t₀))` with

```
Ω₁ = Σₐ [fₐ] H⁽ᵃ⁾
Ω₂ = ½ Σₐᵦ [fₐfᵦ] [H⁽ᵃ⁾, H⁽ᵇ⁾]
Ω₃ = ⅙ Σₐᵦ𝚌 [fₐfᵦf𝚌] ( [[H⁽ᵃ⁾,H⁽ᵇ⁾],H⁽ᶜ⁾] + [H⁽ᵃ⁾,[H⁽ᵇ⁾,H⁽ᶜ⁾]] )
```

```julia
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "magnus", order = 2, nsteps = 100)

# ...or the driver directly
Ω = magnus_generator(channels, t0, t1; order = 2)
ψ = magnus_evolve(
    channels, ψ0, 0.0, 10.0;
    nsteps = 100, order = 2,
    tdvp_kwargs = (; cutoff = 1e-10, maxdim = 128),
)
```

Because the brackets carry the factors of `-i`, `Ω` is anti-Hermitian
and `exp(Ω)` is unitary — so unlike the Dyson driver, `magnus_evolve`
preserves the norm exactly. The exponential is applied as
`tdvp(Ω, 1.0, ψ)`: the generator already carries the whole step, so the
TDVP "time" is 1, matching the paper's exponentiation of `Ω` at `τ = 1`.

The commutator structure is what keeps each term extensive — the
non-overlapping ("disjoint") contributions cancel between `AB` and `BA`.

## Orders

Orders 1–3 are supported; **order 2 is the right choice** (order 3
converges no faster and costs more). See
[Why Dyson improves order-by-order but Magnus doesn't](@ref) for why. A
fourth-order `Ω₄` is not offered: reaching genuine 6th order requires
`Ω₃` and `Ω₄` together in a nested-commutator basis that is nontrivial
to get right, so [CFET](@ref "Commutator-free propagator (CFET)") is
the route to higher order here — it is both cheaper and more extensible,
since commutator-free schemes are specified by coefficient tables rather
than commutator algebra.

## Reference

```@docs
magnus_generator
magnus_terms
magnus_evolve
```
