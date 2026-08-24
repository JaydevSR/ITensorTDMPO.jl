```@meta
CurrentModule = TDVPlus
```

# Driving channels

The Dyson and Magnus constructions both start from the paper's
decomposition of the Hamiltonian into **driving channels**,

```
H(t) = Σₐ fₐ(t) H⁽ᵃ⁾
```

where each `H⁽ᵃ⁾` is a *time-independent* MPO and each `fₐ` a scalar
driving function. This is what makes the expansions tractable: the
operator content (products and commutators of the `H⁽ᵃ⁾`) is computed
once, and all time dependence enters through scalar integrals of the `fₐ`.

```julia
channels = DrivingChannels([(1.0, Hzz), (ramp, Hx)])
channels(0.5)        # assembles H(0.5) as an MPO (useful for testing)
nchannels(channels)  # 2
```

[`time_evolve`](@ref) builds this internally when handed a plain list, so
you only need it explicitly to reuse one decomposition across several
calls.

## Channel specification

Channels are given as a list of `(driving, MPO)` tuples. The MPO and the
driving are told apart by type, so `(H, f)` works as well as `(f, H)`,
and `H => f` pairs are accepted too.

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

## Time-ordered integrals

The scalar prefactors in both the Dyson and Magnus expansions are the
time-ordered integrals

```
[f₁ f₂ … fₙ] = (-i)ⁿ ∫dt₁ ∫^{t₁}dt₂ … ∫^{t_{n-1}}dtₙ  f₁(t₁) f₂(t₂) … fₙ(tₙ)
```

with `t₁ > t₂ > … > tₙ`, so `f₁` is evaluated at the *latest* time. The
factors of `-i` are folded into the bracket.

```julia
time_ordered_integral([f, g], t0, t1)     # [f g]
time_ordered_integral(fill(one, 3), 0, Δ) # ≈ (-i)³ Δ³/3!
```

These are evaluated by repeated cumulative Simpson quadrature on a
uniform grid — `O(n · npoints)` rather than the `O(npointsⁿ)` of naive
nested quadrature. The paper instead uses a quantics tensor-train
representation ([QuanticsTT.jl](https://github.com/VictorVanthilt/QuanticsTT.jl)),
which matters at high order; the grid approach here is simpler and
adequate for the orders implemented.

The implementation satisfies the paper's factoring property
`[fₐfᵦ] + [fᵦfₐ] = [fₐ][fᵦ]`, which is checked in the test suite.

`prefactor` (default `-im`) sets the factor carried per order; pass `-1`
for imaginary time — see [Imaginary time](@ref).

## Reference

```@docs
DrivingChannels
nchannels
identity_mpo
commutator
time_ordered_integral
cumulative_integral
```
