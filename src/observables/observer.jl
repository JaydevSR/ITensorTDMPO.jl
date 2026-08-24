"""
    EvolutionObserver(:name => f, ...)

Accumulates measurements along a time evolution. Each `f` is called on
the state after every step — as `f(state)` or, if it accepts two
arguments, as `f(state, t)` — and the results are collected under
`:name`.

An `EvolutionObserver` is itself the callback the drivers expect, so pass
it as `step_observer!`. Read the accumulated data back with
[`results`](@ref).

Pass `every = k` to record only every `k`-th step. Use this for
diagnostics that cost more than the evolution itself — a DMRG-based gap,
say — so they can be sampled sparsely without slowing the run. An
explicit call to [`observe!`](@ref) always records, regardless of `every`.

# Examples

```julia
obs = EvolutionObserver(
    :Sz      => ψ -> expect(ψ, "Sz"),
    :entropy => ψ -> entanglement_entropy(ψ),
    :chi     => maxlinkdim,
    :energy  => (ψ, t) -> instantaneous_energy(channels, t, ψ),
)

observe!(obs, ψ0, 0.0)          # record the initial state (optional)
ψ = time_evolve(channels, ψ0, 0.0, 10.0; nsteps = 100, step_observer! = obs)

r = results(obs)                # (; step, time, Sz, entropy, chi, energy)
r.time, r.entropy
```
"""
struct EvolutionObserver
    names::Vector{Symbol}
    functions::Vector{Any}
    steps::Vector{Int}
    times::Vector{Float64}
    data::Vector{Vector{Any}}
    every::Int
end

function EvolutionObserver(measurements::Pair...; every::Integer = 1)
    isempty(measurements) &&
        throw(ArgumentError("`EvolutionObserver` needs at least one measurement."))
    every >= 1 || throw(ArgumentError("`every` must be at least 1, got $every."))
    names = Symbol[Symbol(first(m)) for m in measurements]
    if length(unique(names)) != length(names)
        throw(ArgumentError("measurement names must be unique, got $names."))
    end
    fs = Any[last(m) for m in measurements]
    for (n, f) in zip(names, fs)
        applicable(f, MPS()) || applicable(f, MPS(), 0.0) || throw(
            ArgumentError(
                "measurement `$n` is a $(typeof(f)), which is not callable as `f(state)` or `f(state, t)`."
            )
        )
    end
    return EvolutionObserver(names, fs, Int[], Float64[], [Any[] for _ in names], every)
end

EvolutionObserver(measurements::AbstractVector{<:Pair}; kwargs...) =
    EvolutionObserver(measurements...; kwargs...)

_measure(f, state, t) = applicable(f, state, t) ? f(state, t) : f(state)

"""
    observe!(obs::EvolutionObserver, state, t; step = <next>)

Record one measurement row. The drivers call this for you; call it
directly to record the initial state before evolving.
"""
function observe!(obs::EvolutionObserver, state, t; step = length(obs.steps) + 1)
    push!(obs.steps, step)
    push!(obs.times, float(t))
    for (i, f) in enumerate(obs.functions)
        push!(obs.data[i], _measure(f, state, t))
    end
    return obs
end

# The drivers' `step_observer!` calling convention. Steps are skipped
# according to `every`, so that expensive diagnostics can be sampled
# sparsely along a long evolution. An explicit `observe!` always records.
function (obs::EvolutionObserver)(; step, t_start, t_stop, state, kwargs...)
    mod(step, obs.every) == 0 || return obs
    return observe!(obs, state, t_stop; step)
end

"""
    results(obs::EvolutionObserver)

The accumulated measurements as a named tuple of vectors, with `step` and
`time` alongside one entry per measurement. Element types are narrowed,
so `results(obs).entropy` comes back as a `Vector{Float64}` rather than
`Vector{Any}`.
"""
function results(obs::EvolutionObserver)
    narrowed = map(v -> isempty(v) ? v : identity.(v), obs.data)
    return (;
        step = copy(obs.steps),
        time = copy(obs.times),
        NamedTuple{Tuple(obs.names)}(Tuple(narrowed))...,
    )
end

"""
    empty!(obs::EvolutionObserver)

Discard accumulated data, keeping the measurement definitions, so the
observer can be reused for another run.
"""
function Base.empty!(obs::EvolutionObserver)
    empty!(obs.steps)
    empty!(obs.times)
    foreach(empty!, obs.data)
    return obs
end

Base.length(obs::EvolutionObserver) = length(obs.steps)

function Base.show(io::IO, obs::EvolutionObserver)
    return print(
        io, "EvolutionObserver(", join(obs.names, ", "), ") with ",
        length(obs), " recorded step(s)"
    )
end
