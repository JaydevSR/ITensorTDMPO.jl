"""
    magnus_terms(channels::DrivingChannels, t0, t; order, cutoff, maxdim, npoints, prefactor)

The individual terms of the Magnus generator `Ω(t, t0)`, as a vector of
MPOs. Summing them gives [`magnus_generator`](@ref).
"""
function magnus_terms(
        channels::DrivingChannels, t0, t;
        order::Integer,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025,
        prefactor = -im
    )
    1 <= order <= 3 || throw(
        ArgumentError("`order` must be between 1 and 3, got $order.")
    )
    nc = nchannels(channels)
    H = channels.operators
    f = channels.drivings
    bracket(idxs) = time_ordered_integral(
        [f[a] for a in idxs], t0, t; npoints, prefactor
    )
    comm(A, B) = commutator(A, B; cutoff, maxdim)

    # Inner commutators [H^{(a)}, H^{(b)}], memoized on the ordered pair and
    # obtained for a > b by antisymmetry. At third and fourth order each one
    # is needed by many separate terms.
    cache = Dict{Tuple{Int, Int}, MPO}()
    function pair_commutator(a, b)
        a < b && return get!(() -> comm(H[a], H[b]), cache, (a, b))
        return -pair_commutator(b, a)   # a == b is never requested
    end

    terms = MPO[]

    # Ω₁ = Σₐ [fₐ] H^{(a)}
    for a in 1:nc
        coeff = bracket((a,))
        iszero(coeff) && continue
        push!(terms, coeff * H[a])
    end
    order == 1 && return terms

    # Ω₂ = ½ Σₐᵦ [fₐfᵦ] [H^{(a)}, H^{(b)}].
    # The a = b terms vanish, and antisymmetry of the commutator folds
    # (b,a) onto (a,b), recovering the paper's α = ½([fₐfᵦ] - [fᵦfₐ]).
    for a in 1:nc, b in (a + 1):nc
        α = (bracket((a, b)) - bracket((b, a))) / 2
        iszero(α) && continue
        push!(terms, α * pair_commutator(a, b))
    end
    order == 2 && return terms

    # Ω₃ = ⅙ Σₐᵦ𝚌 [fₐfᵦf𝚌] ( [[H^{(a)},H^{(b)}],H^{(c)}] + [H^{(a)},[H^{(b)},H^{(c)}]] )
    # Each nested commutator is folded onto its ordered inner pair.
    κ3(a, b, c) = bracket((a, b, c)) / 6
    for a in 1:nc, b in (a + 1):nc, c in 1:nc
        coeff = κ3(a, b, c) - κ3(b, a, c)
        iszero(coeff) && continue
        push!(terms, coeff * comm(pair_commutator(a, b), H[c]))
    end
    for a in 1:nc, b in 1:nc, c in (b + 1):nc
        coeff = κ3(a, b, c) - κ3(a, c, b)
        iszero(coeff) && continue
        push!(terms, coeff * comm(H[a], pair_commutator(b, c)))
    end
    return terms
end

"""
    magnus_generator(channels::DrivingChannels, t0, t; order = 2, kwargs...)

The Magnus generator `Ω(t, t0)` up to `order`, as an MPO, such that
`U(t, t0) ≈ exp(Ω(t, t0))`.

Following [Vanthilt et al.](https://arxiv.org/abs/2605.21597), writing
`H(t) = Σₐ fₐ(t) H^{(a)}`, each Magnus term is a linear combination of
(nested) commutators of the *time-independent* channel operators, with
scalar prefactors given by the time-ordered integrals of the driving
functions:

```math
\\Omega_1 = \\sum_a [f_a] H^{(a)}, \\qquad
\\Omega_2 = \\tfrac{1}{2} \\sum_{ab} [f_a f_b] \\, [H^{(a)}, H^{(b)}]
```

The commutator structure is what keeps each term extensive: the
non-overlapping ("disjoint") contributions cancel between `AB` and `BA`.
The `[…]` brackets carry the factors of `-i`, so `Ω` is anti-Hermitian
for Hermitian channel operators and real driving functions, and
`exp(Ω)` is unitary.

# Orders

Supported orders are 1 through 3; **2 is the right default**:

| `order` | measured convergence order | note |
|---|---|---|
| 1 | 2 | exponential midpoint-like |
| 2 | **~3.9** | the recommended setting |
| 3 | ~3.8 | no better than 2, and costs more per step |

Orders 2 and 3 coincide because the truncated Magnus expansion is
time-symmetric, and time-symmetric methods have even order — so the odd
term `Ω₃` buys nothing. Reaching sixth order would need `Ω₄`; a partial
implementation of it measured ~4th order rather than the expected 6th and
was removed rather than shipped unverified. For higher order, prefer
[`cfet_evolve`](@ref), which reaches fourth order more cheaply and
extends to higher-order commutator-free schemes.

`prefactor` (default `-im`) sets the factor carried per order; pass `-1`
for imaginary time, which makes `Ω` Hermitian and `exp(Ω)` a
non-unitary contraction.

!!! note "Scope"
    Commutators are formed by direct MPO multiplication rather than
    through the paper's finite-state-machine encoding. See the README.
"""
function magnus_generator(
        channels::DrivingChannels, t0, t;
        order::Integer = 2,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025,
        prefactor = -im
    )
    terms = magnus_terms(channels, t0, t; order, cutoff, maxdim, npoints, prefactor)
    isempty(terms) && return 0.0 * identity_mpo(channels)
    length(terms) == 1 && return only(terms)
    return sum(terms; cutoff, maxdim)
end

"""
    magnus_evolve(channels, ψ0, times; order = 2, kwargs...)
    magnus_evolve(channels, ψ0, t_start, t_stop; dt = nothing, nsteps = nothing, kwargs...)

Time-evolve `ψ0` under the time-dependent Hamiltonian carried by
`channels` by building the Magnus generator `Ω` on each interval of
`times` and applying `exp(Ω)` to the state with TDVP.

For real-time evolution `Ω` is anti-Hermitian, so `exp(Ω)` is unitary and
the norm is preserved. As with the Dyson driver, the time dependence
within each step is captured to the given order rather than frozen.

The exponential is applied as `tdvp(Ω, 1.0, ψ)`: the generator already
carries the full step, so the TDVP "time" is 1, exactly as in Sec. IV of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597) where `Ω` is
exponentiated with a Taylor MPO at `τ = 1`.

# Keywords

  - `order = 2`: order of the Magnus expansion on each step (1–3); see
    [`magnus_generator`](@ref) for what each order buys.
  - `generator_kwargs = (;)`: forwarded to [`magnus_generator`](@ref),
    including `prefactor` for imaginary time.
  - `tdvp_kwargs = (; cutoff = 1e-10)`: forwarded to `tdvp`. Increase
    `nsteps` here if the exponentiation of `Ω` itself needs refining.
  - `normalize = false`: renormalize after each step. Leave `false` for
    real time, where the evolution is already unitary; set `true` for
    imaginary time, where `exp(Ω)` contracts the state.
  - `(step_observer!) = nothing`: callback
    `step_observer!(; step, t_start, t_stop, state)` after each step.
  - `outputlevel = 0`: set `>= 1` to print progress.

Returns the final evolved `MPS`.
"""
function magnus_evolve(
        channels::DrivingChannels, ψ0::MPS, times;
        order::Integer = 2,
        generator_kwargs = (;),
        tdvp_kwargs = (; cutoff = 1.0e-10),
        normalize::Bool = false,
        (step_observer!) = nothing,
        outputlevel = 0
    )
    ψ = copy(ψ0)
    nsteps = length(times) - 1
    for step in 1:nsteps
        t_start = times[step]
        t_stop = times[step + 1]
        Ω = magnus_generator(channels, t_start, t_stop; order, generator_kwargs...)
        ψ = tdvp(Ω, 1.0, ψ; tdvp_kwargs...)
        normalize && LinearAlgebra.normalize!(ψ)
        if !isnothing(step_observer!)
            step_observer!(; step, t_start, t_stop, state = ψ)
        end
        if outputlevel >= 1
            println(
                "Magnus step $step/$nsteps: t = $(round(t_stop; digits = 6)), maxlinkdim = $(maxlinkdim(ψ))"
            )
            flush(stdout)
        end
    end
    return ψ
end

function magnus_evolve(
        channels::DrivingChannels, ψ0::MPS, t_start::Number, t_stop::Number;
        dt = nothing, nsteps = nothing, kwargs...
    )
    return magnus_evolve(channels, ψ0, _time_grid(t_start, t_stop, dt, nsteps); kwargs...)
end

"""
    magnus_evolve([(f1, H1), (f2, H2), ...], ψ0, args...; kwargs...)

Convenience form taking the channels as a plain list of `(driving, MPO)`
tuples or pairs (see [`DrivingChannels`](@ref)), for calling this driver
directly without building the `DrivingChannels` object yourself.
"""
function magnus_evolve(channels::AbstractVector, ψ0::MPS, args...; kwargs...)
    return magnus_evolve(DrivingChannels(channels), ψ0, args...; kwargs...)
end
