"""
    trace_distance(a::MPS, b::MPS)

`sqrt(1 - |⟨a|b⟩|²)` between normalized states, the error measure used by
the adaptive stepper and by the paper's finite-size benchmark. Both
states are normalized internally, so a norm drift from truncation does
not register as an error.
"""
function trace_distance(a::MPS, b::MPS)
    ov = abs(inner(a, b)) / (norm(a) * norm(b))
    return sqrt(max(zero(ov), 1 - min(one(ov), ov)^2))
end

# Local error order (one above the global convergence order) for the
# step-doubling estimate. Used only to choose the next step size.
_local_order(::Algorithm"magnus", order) = order == 1 ? 3 : 5
_local_order(::Algorithm"cfet", order) = order == 2 ? 3 : 5
_local_order(::Algorithm"piecewise_constant", order) = 3
_local_order(::Algorithm"dyson", order) = order + 1
_local_order(::Algorithm, order) = 3

"""
    adaptive_time_evolve(channels, ψ0, t_start, t_stop; tol = 1e-6, kwargs...)

Evolve from `t_start` to `t_stop` choosing each step size automatically,
so that you set an accuracy rather than a step count.

Each candidate step is taken twice — once at size `dt`, and once as two
steps of `dt/2` — and the [`trace_distance`](@ref) between the two
results estimates the local error. The step is accepted when that
estimate falls below `tol`, and the next step size is scaled by
`(tol/err)^(1/p)` with `p` the local error order of the chosen algorithm.
The finer of the two solutions is the one kept.

This costs three sub-steps per accepted step, so it pays off when
`‖Ḣ‖` varies strongly over the evolution — as it does for a ramp that is
fast near a gap minimum and slow elsewhere — and not for a uniform drive.

!!! warning "`tol` has a floor"
    Step-doubling assumes shrinking the step always shrinks the error,
    which requires the per-step integrator itself to be exact. TDVP is
    not: push `tol` far enough below its own per-application roundoff and
    the coarse/fine comparison starts measuring roundoff noise instead of
    integration error, at which point *more* steps make the true result
    *worse*. A telltale sign is `history.errors` reading exactly `0.0`.
    Measured on the README's benchmark model, the true error is minimized
    around `tol ≈ 1e-7` and increases for `tol ≤ 1e-8`; keep `tol` at or
    above roughly the `cutoff` in use, not many orders tighter. See the
    README for the full measurement.

# Keywords

  - `tol = 1e-6`: target local error per step.
  - `dt_init`: first step size (default: a twentieth of the interval).
  - `dt_min`, `dt_max`: bounds on the step size. Falling below `dt_min`
    raises an error rather than stalling.
  - `safety = 0.9`, `grow_max = 5.0`, `shrink_min = 0.2`: step-size
    controller limits.
  - `maxsteps = 100_000`: guard against runaway loops.
  - `(step_observer!) = nothing`: called after each *accepted* step, as
    `step_observer!(; step, t_start, t_stop, state)`.
  - `outputlevel = 0`: `>= 1` prints each accepted step and its error,
    `>= 2` also prints rejections.

All remaining keywords (`alg`, `order`, `cutoff`, `operator_cutoff`, …)
are passed to [`time_evolve`](@ref) for the individual sub-steps.

Returns `(state, history)` where `history` is a named tuple of vectors
`(; times, dts, errors)` recording the accepted steps.
"""
function adaptive_time_evolve(
        channels::DrivingChannels, ψ0::MPS, t_start::Number, t_stop::Number;
        alg = "magnus",
        order = nothing,
        tol = 1.0e-6,
        dt_init = nothing,
        dt_min = nothing,
        dt_max = nothing,
        safety = 0.9,
        grow_max = 5.0,
        shrink_min = 0.2,
        maxsteps::Integer = 100_000,
        (step_observer!) = nothing,
        outputlevel = 0,
        kwargs...
    )
    _check_algorithm(alg)
    total = t_stop - t_start
    iszero(total) && return copy(ψ0), (; times = [float(t_start)], dts = Float64[], errors = Float64[])
    p = _local_order(_algorithm(alg), something(order, 2))

    dt = something(dt_init, total / 20)
    dtmin = something(dt_min, abs(total) * 1.0e-10)
    dtmax = something(dt_max, abs(total))
    sub_kwargs = isnothing(order) ? kwargs : (; order, kwargs...)

    ψ = copy(ψ0)
    t = float(t_start)
    times = [t]
    dts = Float64[]
    errors = Float64[]
    step = 0

    onestep(state, a, b) = time_evolve(channels, state, [a, b]; alg, sub_kwargs...)

    while (t - t_stop) * sign(total) < -eps(float(max(abs(t), abs(t_stop))))
        step > maxsteps && error("adaptive stepping exceeded `maxsteps = $maxsteps`.")
        dt = sign(total) * min(abs(dt), abs(t_stop - t))   # do not overshoot
        coarse = onestep(ψ, t, t + dt)
        half = onestep(ψ, t, t + dt / 2)
        fine = onestep(half, t + dt / 2, t + dt)
        err = trace_distance(coarse, fine)

        if err <= tol || abs(dt) <= dtmin
            step += 1
            t += dt
            ψ = fine
            push!(times, t)
            push!(dts, dt)
            push!(errors, err)
            if !isnothing(step_observer!)
                step_observer!(; step, t_start = t - dt, t_stop = t, state = ψ)
            end
            outputlevel >= 1 && println(
                "adaptive step $step: t = $(round(t; digits = 6)), dt = $(round(dt; sigdigits = 3)), err = $(round(err; sigdigits = 3)), maxlinkdim = $(maxlinkdim(ψ))"
            )
        else
            outputlevel >= 2 && println(
                "  rejected dt = $(round(dt; sigdigits = 3)) (err = $(round(err; sigdigits = 3)) > tol)"
            )
        end

        # Step-size controller. `err == 0` means "as large as allowed".
        factor = iszero(err) ? grow_max : safety * (tol / err)^(1 / p)
        dt *= clamp(factor, shrink_min, grow_max)
        if abs(dt) < dtmin
            error(
                "adaptive stepping needs dt = $(abs(dt)) < dt_min = $dtmin to meet tol = $tol at t = $t; loosen `tol` or lower `dt_min`."
            )
        end
        abs(dt) > dtmax && (dt = sign(total) * dtmax)
    end
    return ψ, (; times, dts, errors)
end
