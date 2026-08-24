"""
    dyson_terms(channels::DrivingChannels, t0, t; order, cutoff, maxdim, npoints)

The individual terms of the Dyson series for `U(t, t0)`, as a vector of
MPOs including the zeroth-order identity. Summing them gives
[`dyson_mpo`](@ref); they are exposed separately so that the truncation
of the sum can be controlled by the caller.
"""
function dyson_terms(
        channels::DrivingChannels, t0, t;
        order::Integer,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025,
        prefactor = -im
    )
    order >= 0 || throw(ArgumentError("`order` must be non-negative, got $order."))
    nc = nchannels(channels)
    terms = MPO[identity_mpo(channels)]
    # `partials` maps each operator string (a₁, …, aₙ) to the MPO product
    # H^{(a₁)} ⋯ H^{(aₙ)}. Only the current order is held at a time; each
    # level is extended from the previous one rather than rebuilt, so the
    # products are shared across the `nc` extensions of each string.
    partials = Pair{Vector{Int}, MPO}[]
    for n in 1:order
        if n == 1
            partials = [[a] => channels.operators[a] for a in 1:nc]
        else
            next = Vector{Pair{Vector{Int}, MPO}}(undef, length(partials) * nc)
            i = 0
            for (combo, P) in partials, b in 1:nc
                next[i += 1] =
                    vcat(combo, b) => apply(P, channels.operators[b]; cutoff, maxdim)
            end
            partials = next
        end
        for (combo, P) in partials
            coeff = time_ordered_integral(
                [channels.drivings[a] for a in combo], t0, t; npoints, prefactor
            )
            iszero(coeff) && continue
            push!(terms, coeff * P)
        end
    end
    return terms
end

"""
    dyson_mpo(channels::DrivingChannels, t0, t; order = 2, kwargs...)

The `order`-th order Dyson series approximation to the time-evolution
operator `U(t, t0)` of the time-dependent Hamiltonian carried by
`channels`, as a single MPO.

Following [Vanthilt et al.](https://arxiv.org/abs/2605.21597), writing
`H(t) = Σₐ fₐ(t) H^{(a)}` the Dyson series reads

```math
U(t, t_0) = 1 + \\sum_a [f_a] H^{(a)}
              + \\sum_{ab} [f_a f_b] H^{(a)} H^{(b)} + …
```

where `[f_{a₁} ⋯ f_{aₙ}]` are the time-ordered integrals of
[`time_ordered_integral`](@ref), which carry the factors of `-i`. The
operator strings are built by direct MPO multiplication and the series is
summed with truncation.

!!! warning "Scope and accuracy"
    This is the *direct* construction: operator strings are formed as
    explicit MPO products. It reproduces the Dyson series of the paper
    term by term, but not the paper's size-extensive finite-state-machine
    encoding, which keeps the bond dimension low and makes the accuracy
    independent of chain length — on finite systems as well as in the
    thermodynamic limit.

    As a result the error here grows steeply with system size, and
    [`dyson_evolve`](@ref) is already less accurate than freezing the
    Hamiltonian by a chain length of about 4 sites. Prefer
    [`magnus_evolve`](@ref), whose exponentiation resums the disjoint
    contributions for free. See the README for measured numbers.

# Keywords

  - `order = 2`: highest order in the time step retained.
  - `cutoff = 1e-12`, `maxdim`: truncation used for the intermediate MPO
    products and for the final sum.
  - `npoints = 1025`: grid resolution for the time-ordered integrals.

A truncated Dyson MPO is *not* exactly unitary, so applying it to a
normalized state does not preserve the norm exactly; the deviation is of
the order of the neglected terms.
"""
function dyson_mpo(
        channels::DrivingChannels, t0, t;
        order::Integer = 2,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025,
        prefactor = -im
    )
    terms = dyson_terms(channels, t0, t; order, cutoff, maxdim, npoints, prefactor)
    length(terms) == 1 && return only(terms)
    return sum(terms; cutoff, maxdim)
end

"""
    dyson_evolve(channels, ψ0, times; order = 2, kwargs...)
    dyson_evolve(channels, ψ0, t_start, t_stop; dt = nothing, nsteps = nothing, kwargs...)

Time-evolve `ψ0` under the time-dependent Hamiltonian carried by
`channels` by building the `order`-th order Dyson MPO on each interval of
`times` and applying it to the state.

Unlike [`piecewise_constant_tdvp`](@ref), the Hamiltonian is *not* frozen
within a step: the time dependence within each interval is captured
exactly, to the given order, through the time-ordered integrals of the
driving functions. This allows substantially larger time steps for
rapidly varying drives.

# Keywords

  - `order = 2`: order of the Dyson expansion on each step.
  - `mpo_kwargs = (;)`: forwarded to [`dyson_mpo`](@ref) (`cutoff`,
    `maxdim`, `npoints`).
  - `apply_kwargs = (; cutoff = 1e-10)`: forwarded to `apply(U, ψ)`.
  - `normalize = true`: renormalize the state after each step. A
    truncated Dyson MPO is not exactly unitary, so for real-time
    evolution this removes the spurious norm drift.
  - `(step_observer!) = nothing`: callback
    `step_observer!(; step, t_start, t_stop, state)` after each step.
  - `outputlevel = 0`: set `>= 1` to print progress.

Returns the final evolved `MPS`.
"""
function dyson_evolve(
        channels::DrivingChannels, ψ0::MPS, times;
        order::Integer = 2,
        mpo_kwargs = (;),
        apply_kwargs = (; cutoff = 1.0e-10),
        normalize::Bool = true,
        (step_observer!) = nothing,
        outputlevel = 0
    )
    ψ = copy(ψ0)
    nsteps = length(times) - 1
    for step in 1:nsteps
        t_start = times[step]
        t_stop = times[step + 1]
        U = dyson_mpo(channels, t_start, t_stop; order, mpo_kwargs...)
        ψ = apply(U, ψ; apply_kwargs...)
        normalize && LinearAlgebra.normalize!(ψ)
        if !isnothing(step_observer!)
            step_observer!(; step, t_start, t_stop, state = ψ)
        end
        if outputlevel >= 1
            println(
                "Dyson step $step/$nsteps: t = $(round(t_stop; digits = 6)), maxlinkdim = $(maxlinkdim(ψ))"
            )
            flush(stdout)
        end
    end
    return ψ
end

function dyson_evolve(
        channels::DrivingChannels, ψ0::MPS, t_start::Number, t_stop::Number;
        dt = nothing, nsteps = nothing, kwargs...
    )
    return dyson_evolve(channels, ψ0, _time_grid(t_start, t_stop, dt, nsteps); kwargs...)
end

"""
    dyson_evolve([(f1, H1), (f2, H2), ...], ψ0, args...; kwargs...)

Convenience form taking the channels as a plain list of `(driving, MPO)`
tuples or pairs (see [`DrivingChannels`](@ref)), for calling this driver
directly without building the `DrivingChannels` object yourself.
"""
function dyson_evolve(channels::AbstractVector, ψ0::MPS, args...; kwargs...)
    return dyson_evolve(DrivingChannels(channels), ψ0, args...; kwargs...)
end
