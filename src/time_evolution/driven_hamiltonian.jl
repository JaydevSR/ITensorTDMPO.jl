"""
    DrivenHamiltonian(H0, Ht; combine_kwargs = (; alg = "directsum"))

Bundles a time-independent MPO `H0` with a function `Ht(t) -> MPO` giving
the time-dependent part of the Hamiltonian. Calling `H(t)` returns the
total Hamiltonian at time `t` as an MPO, `H0 + Ht(t)`.

`H0` can be `nothing` if the full Hamiltonian is time dependent, in which
case `H(t) === Ht(t)`.

`combine_kwargs` are passed to `+(H0, Ht(t); combine_kwargs...)` and default
to an exact (non-truncating) direct sum, appropriate for Hamiltonian MPOs
which typically already have a small bond dimension.
"""
struct DrivenHamiltonian{H0 <: Union{MPO, Nothing}, F, K <: NamedTuple}
    H0::H0
    Ht::F
    combine_kwargs::K
end

function DrivenHamiltonian(H0, Ht; combine_kwargs = (; alg = "directsum"))
    return DrivenHamiltonian(H0, Ht, combine_kwargs)
end

(H::DrivenHamiltonian)(t) = drive_at(H, t)

drive_at(H::DrivenHamiltonian{Nothing}, t) = H.Ht(t)
function drive_at(H::DrivenHamiltonian, t)
    return +(H.H0, H.Ht(t); H.combine_kwargs...)
end
