using ITensors: ITensors
using ITensorMPS: dmrg, linkind, orthogonalize, siteind

"""
    entanglement_entropy(ψ::MPS, b::Integer = length(ψ) ÷ 2; base = ℯ)

Von Neumann entanglement entropy of the bipartition across bond `b`
(between sites `b` and `b+1`). The Schmidt values are renormalized
internally, so an unnormalized `ψ` is handled correctly.

Pass `base = 2` for entropy in bits.
"""
function entanglement_entropy(ψ::MPS, b::Integer = length(ψ) ÷ 2; base = ℯ)
    if !(1 <= b < length(ψ))
        throw(
            ArgumentError("bond $b is out of range for an MPS of length $(length(ψ)).")
        )
    end
    ϕ = orthogonalize(ψ, b)
    linds = b == 1 ? (siteind(ϕ, b),) : (linkind(ϕ, b - 1), siteind(ϕ, b))
    _, S, _ = ITensors.svd(ϕ[b], linds...)
    d = ITensors.dim(S, 1)
    total = sum(S[n, n]^2 for n in 1:d)
    entropy = zero(float(real(total)))
    for n in 1:d
        p = S[n, n]^2 / total
        p > 0 && (entropy -= p * log(base, p))
    end
    return entropy
end

"""
    entanglement_profile(ψ::MPS; base = ℯ)

[`entanglement_entropy`](@ref) across every bond, as a vector of length
`length(ψ) - 1`.
"""
function entanglement_profile(ψ::MPS; base = ℯ)
    return [entanglement_entropy(ψ, b; base) for b in 1:(length(ψ) - 1)]
end

"""
    instantaneous_energy(channels::DrivingChannels, t, ψ::MPS; cutoff, maxdim)

The expectation value `⟨ψ|H(t)|ψ⟩ / ⟨ψ|ψ⟩` of the instantaneous
Hamiltonian.
"""
function instantaneous_energy(
        channels::DrivingChannels, t, ψ::MPS;
        cutoff = 1.0e-12, maxdim = typemax(Int)
    )
    H = channels(t; cutoff, maxdim)
    return real(inner(ψ', H, ψ)) / real(inner(ψ, ψ))
end

"""
    energy_variance(channels::DrivingChannels, t, ψ::MPS; cutoff, maxdim)

`⟨H(t)²⟩ - ⟨H(t)⟩²`, which vanishes exactly when `ψ` is an eigenstate of
`H(t)`.

This is the cheap adiabaticity diagnostic: it needs one MPO application
and no ground-state solve, and it tells you directly how far the state has
drifted from an instantaneous eigenstate during a ramp. `H²` is never
formed — the variance is computed from `H|ψ⟩`.
"""
function energy_variance(
        channels::DrivingChannels, t, ψ::MPS;
        cutoff = 1.0e-12, maxdim = typemax(Int)
    )
    H = channels(t; cutoff, maxdim)
    Hψ = apply(H, ψ; cutoff, maxdim)
    n2 = real(inner(ψ, ψ))
    e = real(inner(ψ, Hψ)) / n2
    e2 = real(inner(Hψ, Hψ)) / n2
    return e2 - e^2
end

"""
    instantaneous_spectrum(channels, t, ψ_guess; nlevels = 2, weight = 10.0, dmrg_kwargs...)

The lowest `nlevels` eigenstates of `H(t)`, via DMRG, returned as
`(energies, states)`. Excited states are found by penalizing overlap with
the already-converged ones (`weight`).

This runs a full DMRG solve per call and is by far the most expensive
diagnostic here — use it on a coarse subset of times, not every step.
`dmrg_kwargs` are forwarded to `ITensorMPS.dmrg` (`nsweeps`, `maxdim`,
`cutoff`, `outputlevel`, …).
"""
function instantaneous_spectrum(
        channels::DrivingChannels, t, ψ_guess::MPS;
        nlevels::Integer = 2,
        weight = 10.0,
        operator_cutoff = 1.0e-12,
        operator_maxdim = typemax(Int),
        nsweeps = 10,
        outputlevel = 0,
        dmrg_kwargs...
    )
    nlevels >= 1 || throw(ArgumentError("`nlevels` must be at least 1, got $nlevels."))
    H = channels(t; cutoff = operator_cutoff, maxdim = operator_maxdim)
    energies = Float64[]
    states = MPS[]
    for _ in 1:nlevels
        E, ψ = if isempty(states)
            dmrg(H, copy(ψ_guess); nsweeps, outputlevel, dmrg_kwargs...)
        else
            dmrg(H, states, copy(ψ_guess); nsweeps, outputlevel, weight, dmrg_kwargs...)
        end
        push!(energies, real(E))
        push!(states, ψ)
    end
    return energies, states
end

"""
    instantaneous_gap(channels, t, ψ_guess; kwargs...)

The gap `E₁(t) - E₀(t)` of the instantaneous Hamiltonian. Built on
[`instantaneous_spectrum`](@ref), and just as expensive.

The gap is what sets the adiabatic time scale: the sweep must be slow
compared with `1/Δ²` near a gap minimum, so this is the quantity that
tells you where a ramp needs to slow down.
"""
function instantaneous_gap(channels::DrivingChannels, t, ψ_guess::MPS; kwargs...)
    energies, _ = instantaneous_spectrum(channels, t, ψ_guess; nlevels = 2, kwargs...)
    return energies[2] - energies[1]
end

"""
    adiabatic_report(channels, t, ψ; ψ_guess = ψ, gap = false, kwargs...)

A bundle of adiabaticity diagnostics at time `t`, as a named tuple:

  - `energy` — `⟨H(t)⟩`
  - `variance` — `⟨H²⟩ - ⟨H⟩²`, zero for an exact eigenstate
  - `excess` — `⟨H(t)⟩ - E₀(t)`, the energy above the instantaneous
    ground state (`missing` unless `gap = true`)
  - `gap` — `E₁(t) - E₀(t)` (`missing` unless `gap = true`)
  - `fidelity` — `|⟨ψ₀(t)|ψ⟩|` against the instantaneous ground state
    (`missing` unless `gap = true`)

The `gap = false` default keeps this cheap: only `energy` and `variance`
are computed, both without a ground-state solve. Setting `gap = true`
adds a two-level DMRG solve.
"""
function adiabatic_report(
        channels::DrivingChannels, t, ψ::MPS;
        ψ_guess::MPS = ψ,
        gap::Bool = false,
        operator_cutoff = 1.0e-12,
        operator_maxdim = typemax(Int),
        kwargs...
    )
    energy = instantaneous_energy(
        channels, t, ψ; cutoff = operator_cutoff, maxdim = operator_maxdim
    )
    variance = energy_variance(
        channels, t, ψ; cutoff = operator_cutoff, maxdim = operator_maxdim
    )
    if !gap
        return (; t, energy, variance, excess = missing, gap = missing, fidelity = missing)
    end
    energies, states = instantaneous_spectrum(
        channels, t, ψ_guess; nlevels = 2, operator_cutoff, operator_maxdim, kwargs...
    )
    fidelity = abs(inner(states[1], ψ)) / (norm(states[1]) * norm(ψ))
    return (;
        t, energy, variance,
        excess = energy - energies[1],
        gap = energies[2] - energies[1],
        fidelity,
    )
end
