```@meta
CurrentModule = ITensorTDMPO
```

# Adaptive stepping

Set an accuracy instead of a step count:

```julia
ψ = time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", adaptive = true, tol = 1e-7)

# adaptive_time_evolve also returns the step history
ψ, hist = adaptive_time_evolve(channels, ψ0, 0.0, 10.0; alg = "cfet", tol = 1e-7)
hist.dts, hist.errors      # where the stepper had to slow down
```

Each candidate step is taken once at `dt` and again as two steps of
`dt/2`; the [`trace_distance`](@ref) between the results estimates the
local error, the step is accepted when that is below `tol`, and the
next step size is scaled by `(tol/err)^(1/p)`. This costs three
sub-steps per accepted step, so it pays off when `‖Ḣ‖` varies strongly
across the evolution — a ramp that must crawl through a gap minimum and
can sprint elsewhere — and not for a uniform drive.

!!! warning "`tol` has a floor too"
    Step-doubling assumes the per-step integrator is otherwise exact, so
    that shrinking the step always shrinks the error. TDVP is not exact:
    push `tol` far enough below its own per-application roundoff and the
    coarse/fine comparison stops measuring the integration error at all
    — it measures roundoff noise instead. Measured on the benchmark
    model on [The `time_evolve` interface](@ref) page, the true error
    (against an independent reference) is minimized around `tol ≈ 1e-7`
    and *increases* for `tol = 1e-8, 1e-9, 1e-10`, even though the
    stepper dutifully takes more steps each time:

    | `tol` | true error | steps |
    |---|---|---|
    | 1e-5 | 7.4e-7 | 6 |
    | 1e-6 | 3.8e-7 | 8 |
    | **1e-7** | **3.4e-7 (best)** | 12 |
    | 1e-8 | 8.7e-7 | 21 |
    | 1e-10 | 1.3e-6 | 34 |

    A telltale sign you've crossed this floor: the recorded step errors
    in `hist.errors` start reading exactly `0.0`. Keep `tol` at or above
    roughly the state/operator `cutoff` in use, not many orders tighter.

## Reference

```@docs
adaptive_time_evolve
trace_distance
```
