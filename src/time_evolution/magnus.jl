"""
    magnus_terms(channels::DrivingChannels, t0, t; order, cutoff, maxdim, npoints)

The individual terms of the Magnus generator `Ω(t, t0)`, as a vector of
MPOs. Summing them gives [`magnus_generator`](@ref).
"""
function magnus_terms(
        channels::DrivingChannels, t0, t;
        order::Integer,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025
    )
    1 <= order <= 3 || throw(
        ArgumentError("`order` must be 1, 2 or 3, got $order.")
    )
    nc = nchannels(channels)
    H = channels.operators
    f = channels.drivings
    bracket(idxs) = time_ordered_integral([f[a] for a in idxs], t0, t; npoints)
    terms = MPO[]

    # Ω₁ = Σₐ [fₐ] H^{(a)}
    for a in 1:nc
        coeff = bracket((a,))
        iszero(coeff) && continue
        push!(terms, coeff * H[a])
    end
    order == 1 && return terms

    # Ω₂ = ½ Σₐᵦ [fₐfᵦ] [H^{(a)}, H^{(b)}]
    for a in 1:nc, b in 1:nc
        a == b && continue          # [H^{(a)}, H^{(a)}] = 0
        coeff = bracket((a, b)) / 2
        iszero(coeff) && continue
        push!(terms, coeff * commutator(H[a], H[b]; cutoff, maxdim))
    end
    order == 2 && return terms

    # Ω₃ = ⅙ Σₐᵦ𝚌 [fₐfᵦf𝚌] ( [[H^{(a)},H^{(b)}],H^{(c)}] + [H^{(a)},[H^{(b)},H^{(c)}]] )
    for a in 1:nc, b in 1:nc, c in 1:nc
        coeff = bracket((a, b, c)) / 6
        iszero(coeff) && continue
        if a != b               # otherwise [[H^{(a)},H^{(b)}],·] vanishes
            inner_ab = commutator(H[a], H[b]; cutoff, maxdim)
            push!(terms, coeff * commutator(inner_ab, H[c]; cutoff, maxdim))
        end
        if b != c               # otherwise [·,[H^{(b)},H^{(c)}]] vanishes
            inner_bc = commutator(H[b], H[c]; cutoff, maxdim)
            push!(terms, coeff * commutator(H[a], inner_bc; cutoff, maxdim))
        end
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

```math
\\Omega_3 = \\tfrac{1}{6} \\sum_{abc} [f_a f_b f_c] \\left(
    [[H^{(a)},H^{(b)}],H^{(c)}] + [H^{(a)},[H^{(b)},H^{(c)}]] \\right)
```

The commutator structure is what keeps each term extensive: the
non-overlapping ("disjoint") contributions cancel between `AB` and `BA`.
The `[…]` brackets carry the factors of `-i`, so `Ω` is anti-Hermitian
for Hermitian channel operators and real driving functions, and
`exp(Ω)` is unitary.

Supported orders are 1, 2 and 3.

!!! note "Scope"
    As with [`dyson_mpo`](@ref), commutators are formed by direct MPO
    multiplication rather than through the paper's finite-state-machine
    encoding. See the README.
"""
function magnus_generator(
        channels::DrivingChannels, t0, t;
        order::Integer = 2,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025
    )
    terms = magnus_terms(channels, t0, t; order, cutoff, maxdim, npoints)
    isempty(terms) && return 0.0 * identity_mpo(channels)
    length(terms) == 1 && return only(terms)
    return +(terms...; cutoff, maxdim)
end

"""
    magnus_evolve(channels, ψ0, times; order = 2, kwargs...)
    magnus_evolve(channels, ψ0, t_start, t_stop; dt = nothing, nsteps = nothing, kwargs...)

Time-evolve `ψ0` under the time-dependent Hamiltonian carried by
`channels` by building the Magnus generator `Ω` on each interval of
`times` and applying `exp(Ω)` to the state with TDVP.

Because `Ω` is anti-Hermitian, `exp(Ω)` is unitary and this evolution
preserves the norm — unlike [`dyson_evolve`](@ref), where the truncated
series is only approximately unitary. As with the Dyson driver, the time
dependence within each step is captured to the given order rather than
frozen.

The exponential is applied as `tdvp(Ω, 1.0, ψ)`: the generator already
carries the full step, so the TDVP "time" is 1, exactly as in Sec. IV of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597) where `Ω` is
exponentiated with a Taylor MPO at `τ = 1`.

# Keywords

  - `order = 2`: order of the Magnus expansion on each step (1, 2 or 3).
  - `generator_kwargs = (;)`: forwarded to [`magnus_generator`](@ref).
  - `tdvp_kwargs = (; cutoff = 1e-10)`: forwarded to `tdvp`. Increase
    `nsteps` here if the exponentiation of `Ω` itself needs refining.
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
