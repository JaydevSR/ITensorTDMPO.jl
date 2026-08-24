using ITensors: Algorithm, @Algorithm_str

default_state_cutoff() = 1.0e-10
default_operator_cutoff() = 1.0e-12

"""
    EVOLUTION_ALGORITHMS

The algorithm names accepted by [`time_evolve`](@ref).
"""
const EVOLUTION_ALGORITHMS = ("magnus", "cfet", "piecewise_constant", "dyson")

_algorithm(alg::Algorithm) = alg
_algorithm(alg::AbstractString) = Algorithm(String(alg))
_algorithm(alg::Symbol) = Algorithm(String(alg))

# `String(::Algorithm)` is not defined for the parameterisation used here,
# so read the name off the type parameter instead. Doing this wrong turns a
# helpful error into a `MethodError` raised while building the message.
_alg_string(alg::AbstractString) = String(alg)
_alg_string(alg::Symbol) = String(alg)
_alg_string(alg::Algorithm) = string(typeof(alg).parameters[1])

function _check_algorithm(alg)
    name = _alg_string(alg)
    if !(name in EVOLUTION_ALGORITHMS)
        throw(
            ArgumentError(
                "unknown time-evolution algorithm \"$name\"; expected one of " *
                    join(map(a -> "\"$a\"", EVOLUTION_ALGORITHMS), ", ") * "."
            )
        )
    end
    return nothing
end

"""
    time_evolve(channels::DrivingChannels, ψ0::MPS, times; alg = "magnus", kwargs...)
    time_evolve(channels, ψ0, t_start, t_stop; dt = nothing, nsteps = nothing, alg = "magnus", kwargs...)

Evolve `ψ0` under the time-dependent Hamiltonian carried by `channels`,
choosing the integrator with `alg`. This is the single entry point for
every method in the package: the Hamiltonian is always described the same
way — as a [`DrivingChannels`](@ref) decomposition `H(t) = Σₐ fₐ(t) H^{(a)}`
— and only `alg` changes.

# Algorithms

  - `"magnus"` (default) — [`magnus_evolve`](@ref). Expands the step in the
    Magnus series and applies `exp(Ω)`. Unitary, ~4th order, and the
    recommended choice.
  - `"piecewise_constant"` — [`piecewise_constant_tdvp`](@ref). Freezes
    `H(t)` on each interval and runs a TDVP sweep. 2nd order.
  - `"dyson"` — [`dyson_evolve`](@ref). Expands the step in the Dyson
    series and applies the resulting MPO. See the README: its accuracy
    degrades with chain length, so prefer `"magnus"`.

`alg` accepts a `String`, a `Symbol`, or an `ITensors.Algorithm`.

# Keywords

  - `order`: expansion order, for `"magnus"` (1–3) and `"dyson"`. Defaults
    to 2. Passing it with `alg = "piecewise_constant"` is an error, since
    that method does not expand the step.
  - `cutoff = 1e-10`, `maxdim`: truncation of the evolving **state**.
  - `operator_cutoff = 1e-12`, `operator_maxdim`: truncation of the
    **operators** built along the way — the Magnus generator, the Dyson
    MPO, or the frozen `H(t)`.
  - `(step_observer!) = nothing`: callback
    `step_observer!(; step, t_start, t_stop, state)` after each step.
  - `outputlevel = 0`: set `>= 1` to print progress.
  - `alg_kwargs = (;)`: forwarded verbatim to the underlying driver, for
    options specific to one method (`schedule`, `eval_at` and
    `generator_prefactor` for `"piecewise_constant"`; `normalize` for
    `"dyson"`; `npoints` via the operator builders).

The time grid is given either as `times` directly, or as `t_start,
t_stop` with `dt` or `nsteps`, following the same conventions as
`ITensorMPS.tdvp`.

# Examples

```julia
sites = siteinds("S=1/2", 20)
ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

# Channels can be given inline as (driving function, MPO) tuples.
ψ = time_evolve([(one, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                nsteps = 100, cutoff = 1e-10, maxdim = 128)

# Same Hamiltonian, different integrator — nothing else changes.
ψ_pc = time_evolve([(one, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                   alg = "piecewise_constant", nsteps = 100)
```
"""
function time_evolve(channels::DrivingChannels, ψ0::MPS, times; alg = "magnus", kwargs...)
    _check_algorithm(alg)
    return time_evolve(_algorithm(alg), channels, ψ0, times; kwargs...)
end

"""
    time_evolve([(f1, H1), (f2, H2), ...], ψ0, t_start, t_stop; kwargs...)

Convenience form taking the channels as a plain list of
`(driving function, MPO)` tuples (or `H => f` pairs, in either order); the
[`DrivingChannels`](@ref) object is built internally. Construct one
explicitly only if you want to reuse it across several calls.
"""
function time_evolve(channels::AbstractVector, ψ0::MPS, args...; kwargs...)
    return time_evolve(DrivingChannels(channels), ψ0, args...; kwargs...)
end

function time_evolve(
        channels::DrivingChannels, ψ0::MPS, t_start::Number, t_stop::Number;
        dt = nothing, nsteps = nothing, adaptive::Bool = false, kwargs...
    )
    if adaptive
        state, _ = adaptive_time_evolve(
            channels, ψ0, t_start, t_stop; dt_init = dt, kwargs...
        )
        return state
    end
    return time_evolve(channels, ψ0, _time_grid(t_start, t_stop, dt, nsteps); kwargs...)
end

function time_evolve(alg::Algorithm, channels::DrivingChannels, ψ0::MPS, times; kwargs...)
    _check_algorithm(alg)   # throws for an unrecognised name
    return throw(
        ArgumentError(
            "time-evolution algorithm \"$(_alg_string(alg))\" is listed in `EVOLUTION_ALGORITHMS` but has no implementation."
        )
    )
end

function time_evolve(
        ::Algorithm"piecewise_constant", channels::DrivingChannels, ψ0::MPS, times;
        order = nothing,
        cutoff = default_state_cutoff(), maxdim = typemax(Int),
        operator_cutoff = default_operator_cutoff(), operator_maxdim = typemax(Int),
        generator_prefactor = -im, normalize = nothing,
        (step_observer!) = nothing, outputlevel = 0, alg_kwargs = (;)
    )
    if !isnothing(order)
        throw(
            ArgumentError(
                "`order` is not meaningful for `alg = \"piecewise_constant\"`, which freezes `H(t)` on each interval rather than expanding it. Use `alg = \"magnus\"`, `\"cfet\"` or `\"dyson\"` to choose an expansion order."
            )
        )
    end
    return piecewise_constant_tdvp(
        channels, ψ0, times;
        operator_cutoff, operator_maxdim, generator_prefactor,
        tdvp_kwargs = (; cutoff, maxdim),
        normalize = something(normalize, false),
        step_observer!, outputlevel, alg_kwargs...
    )
end

function time_evolve(
        ::Algorithm"magnus", channels::DrivingChannels, ψ0::MPS, times;
        order = 2,
        cutoff = default_state_cutoff(), maxdim = typemax(Int),
        operator_cutoff = default_operator_cutoff(), operator_maxdim = typemax(Int),
        generator_prefactor = -im, normalize = nothing,
        (step_observer!) = nothing, outputlevel = 0, alg_kwargs = (;)
    )
    return magnus_evolve(
        channels, ψ0, times;
        order,
        generator_kwargs = (;
            cutoff = operator_cutoff, maxdim = operator_maxdim,
            prefactor = generator_prefactor,
        ),
        tdvp_kwargs = (; cutoff, maxdim),
        normalize = something(normalize, false),
        step_observer!, outputlevel, alg_kwargs...
    )
end

function time_evolve(
        ::Algorithm"cfet", channels::DrivingChannels, ψ0::MPS, times;
        order = 4,
        cutoff = default_state_cutoff(), maxdim = typemax(Int),
        operator_cutoff = default_operator_cutoff(), operator_maxdim = typemax(Int),
        generator_prefactor = -im, normalize = nothing,
        (step_observer!) = nothing, outputlevel = 0, alg_kwargs = (;)
    )
    return cfet_evolve(
        channels, ψ0, times;
        order, generator_prefactor,
        operator_kwargs = (; cutoff = operator_cutoff, maxdim = operator_maxdim),
        tdvp_kwargs = (; cutoff, maxdim),
        normalize = something(normalize, false),
        step_observer!, outputlevel, alg_kwargs...
    )
end

function time_evolve(
        ::Algorithm"dyson", channels::DrivingChannels, ψ0::MPS, times;
        order = 2,
        cutoff = default_state_cutoff(), maxdim = typemax(Int),
        operator_cutoff = default_operator_cutoff(), operator_maxdim = typemax(Int),
        generator_prefactor = -im, normalize = nothing,
        (step_observer!) = nothing, outputlevel = 0, alg_kwargs = (;)
    )
    return dyson_evolve(
        channels, ψ0, times;
        order,
        mpo_kwargs = (;
            cutoff = operator_cutoff, maxdim = operator_maxdim,
            prefactor = generator_prefactor,
        ),
        apply_kwargs = (; cutoff, maxdim),
        normalize = something(normalize, true),
        step_observer!, outputlevel, alg_kwargs...
    )
end
