function _time_grid(t_start, t_stop, dt::Nothing, nsteps::Nothing)
    return _time_grid(t_start, t_stop, dt, 1)
end
function _time_grid(t_start, t_stop, dt::Nothing, nsteps)
    return range(t_start, t_stop; length = nsteps + 1)
end
function _time_grid(t_start, t_stop, dt, nsteps::Nothing)
    nsteps_float = (t_stop - t_start) / dt
    nsteps_rounded = round(Int, nsteps_float)
    if nsteps_float ≉ nsteps_rounded
        error(
            "(t_stop - t_start) / dt = $(t_stop - t_start) / $dt = $nsteps_float must be an integer."
        )
    end
    return range(t_start, t_stop; length = nsteps_rounded + 1)
end
function _time_grid(t_start, t_stop, dt, nsteps)
    if dt * nsteps ≠ t_stop - t_start
        error(
            "`dt * nsteps == t_stop - t_start` must hold, got dt * nsteps = $dt * $nsteps = $(dt * nsteps) and t_stop - t_start = $(t_stop - t_start)."
        )
    end
    return range(t_start, t_stop; length = nsteps + 1)
end

"""
    piecewise_constant_tdvp(H0, Ht, ψ0, times; kwargs...)
    piecewise_constant_tdvp(H0, Ht, ψ0, t_start, t_stop; dt=nothing, nsteps=nothing, kwargs...)

Time-evolve `ψ0` under the time-dependent Hamiltonian `H(t) = H0 + Ht(t)`
(or just `Ht(t)` if `H0 === nothing`) using TDVP, approximating `H(t)` as
piecewise constant on each interval of `times` (or of the regular grid
from `t_start` to `t_stop` built from `dt`/`nsteps`, following the same
conventions as `ITensorMPS.tdvp`).

This is the simplest general driver for adiabatic ramps and other
time-dependent Hamiltonians: `H(t)` is frozen at one evaluation point per
interval and the state is evolved across that interval with an ordinary
(time-independent) TDVP sweep.

# Arguments

  - `H0::Union{MPO,Nothing}`: the time-independent part of the
    Hamiltonian, or `nothing` if the full Hamiltonian is time dependent.
  - `Ht`: a function `Ht(t) -> MPO` giving the time-dependent part of the
    Hamiltonian (or the full Hamiltonian, if `H0 === nothing`).
  - `ψ0::MPS`: the initial state.
  - `times`: the interval endpoints of the piecewise-constant time grid.

# Keywords

  - `schedule = default_tdvp_schedule`: a function `schedule(t, ψ) ->
    TDVPStepSpec` calibrating, for the interval evaluated at time `t`
    with current state `ψ`, whether to use 1-site or 2-site TDVP and
    whether to perform a global Krylov subspace expansion beforehand
    (needed to grow the bond dimension under 1-site TDVP). See
    [`TDVPStepSpec`](@ref).
  - `eval_at = :midpoint`: whether to freeze `H(t)` at the `:midpoint` or
    the `:start` of each interval.
  - `generator_prefactor = -im`: each step evolves the state by
    `exp(generator_prefactor * dt * H(t))`, matching the convention of
    `ITensorMPS.tdvp`. The default `-im` gives real-time Schrödinger
    evolution; pass `-1` for imaginary-time evolution.
  - `tdvp_kwargs = (;)`: keyword arguments forwarded to
    `ITensorMPS.tdvp` on every step, e.g. `cutoff`, `maxdim`,
    `updater_backend`, `order`.
  - `combine_kwargs = (; alg = "directsum")`: keyword arguments used to
    combine `H0` and `Ht(t)`, forwarded to `+(H0, Ht(t); combine_kwargs...)`.
  - `(step_observer!) = nothing`: an optional callback
    `step_observer!(; step, t_start, t_stop, state)` invoked after each
    interval.
  - `outputlevel = 0`: set to `>= 1` to print progress after each step.

Returns the final evolved `MPS`.
"""
function piecewise_constant_tdvp(
        H0::Union{MPO, Nothing},
        Ht,
        ψ0::MPS,
        times;
        schedule = default_tdvp_schedule,
        eval_at::Symbol = :midpoint,
        generator_prefactor = -im,
        tdvp_kwargs = (;),
        combine_kwargs = (; alg = "directsum"),
        normalize::Bool = false,
        (step_observer!) = nothing,
        outputlevel = 0
    )
    eval_at in (:midpoint, :start) ||
        error("`eval_at` must be `:midpoint` or `:start`, got `$eval_at`.")
    drive = DrivenHamiltonian(H0, Ht; combine_kwargs)
    ψ = copy(ψ0)
    nsteps = length(times) - 1
    for step in 1:nsteps
        t_start = times[step]
        t_stop = times[step + 1]
        dt = t_stop - t_start
        t_eval = eval_at === :midpoint ? (t_start + t_stop) / 2 : t_start
        step_spec = schedule(t_eval, ψ)
        H = drive(t_eval)
        if step_spec.expand_krylov
            ψ = expand(ψ, H; alg = "global_krylov", step_spec.expand_kwargs...)
        end
        ψ = tdvp(
            H, generator_prefactor * dt, ψ;
            nsite = step_spec.nsite, time_start = t_start, tdvp_kwargs...
        )
        normalize && LinearAlgebra.normalize!(ψ)
        if !isnothing(step_observer!)
            step_observer!(; step, t_start, t_stop, state = ψ)
        end
        if outputlevel >= 1
            println(
                "Step $step/$nsteps: t = $(round(t_stop; digits = 6)), nsite = $(step_spec.nsite), maxlinkdim = $(maxlinkdim(ψ))"
            )
            flush(stdout)
        end
    end
    return ψ
end

function piecewise_constant_tdvp(
        H0::Union{MPO, Nothing}, Ht, ψ0::MPS, t_start::Number, t_stop::Number;
        dt = nothing, nsteps = nothing, kwargs...
    )
    times = _time_grid(t_start, t_stop, dt, nsteps)
    return piecewise_constant_tdvp(H0, Ht, ψ0, times; kwargs...)
end

"""
    piecewise_constant_tdvp(channels::DrivingChannels, ψ0::MPS, times; kwargs...)
    piecewise_constant_tdvp(channels, ψ0, t_start, t_stop; dt, nsteps, kwargs...)

Evolve `ψ0` with `H(t)` frozen on each interval, taking the Hamiltonian
from a [`DrivingChannels`](@ref) decomposition rather than an
`H0`/`Ht` pair. This is the form shared with [`dyson_evolve`](@ref) and
[`magnus_evolve`](@ref), so the same Hamiltonian description drives every
method; see [`time_evolve`](@ref) for the common entry point.

`operator_cutoff`/`operator_maxdim` control the truncation used when
assembling `H(t) = Σₐ fₐ(t) H^{(a)}` at each evaluation point; all other
keywords are as for the `H0`/`Ht` method above.
"""
function piecewise_constant_tdvp(
        channels::DrivingChannels, ψ0::MPS, times;
        operator_cutoff = 1.0e-12, operator_maxdim = typemax(Int), kwargs...
    )
    Ht = t -> channels(t; cutoff = operator_cutoff, maxdim = operator_maxdim)
    return piecewise_constant_tdvp(nothing, Ht, ψ0, times; kwargs...)
end

function piecewise_constant_tdvp(
        channels::DrivingChannels, ψ0::MPS, t_start::Number, t_stop::Number;
        dt = nothing, nsteps = nothing, kwargs...
    )
    times = _time_grid(t_start, t_stop, dt, nsteps)
    return piecewise_constant_tdvp(channels, ψ0, times; kwargs...)
end
