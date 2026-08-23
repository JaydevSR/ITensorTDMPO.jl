using ITensorMPS
using ITensors
using LinearAlgebra

# Independent dense-matrix reference machinery, used to check the MPO
# constructions against exact linear algebra on small systems.

"""Dense matrix of an MPO, rows indexed by the primed (output) sites."""
function dense_matrix(H::MPO, sites)
    T = contract(H)
    A = Array(T, [prime.(sites); sites]...)
    d = prod(dim.(sites))
    return reshape(A, d, d)
end

"""Dense state vector of an MPS, using the same index ordering."""
function dense_vector(ψ::MPS, sites)
    T = contract(ψ)
    return reshape(Array(T, sites...), :)
end

"""
Reference time-ordered evolution by a product of midpoint-frozen
exponentials. Converges to the exact propagator as `nsub → ∞`
(second order in the substep), and is completely independent of the
Dyson/Magnus code under test.
"""
function reference_evolution(Hmats, drivings, v0, t0, t1; nsub = 8000)
    dt = (t1 - t0) / nsub
    v = complex(copy(v0))
    for k in 1:nsub
        tmid = t0 + (k - 0.5) * dt
        H = sum(drivings[a](tmid) * Hmats[a] for a in eachindex(Hmats))
        v = exp(-im * dt * Matrix(H)) * v
    end
    return v
end

"""Infidelity 1 - |⟨a|b⟩| between two normalized dense vectors."""
function infidelity(a, b)
    return 1 - abs(dot(a, b)) / (norm(a) * norm(b))
end
