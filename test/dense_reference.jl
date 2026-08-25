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

# `-i H(t) v`, accumulated one channel at a time. Forming `H(t)` as a
# dense matrix first and then applying it would cost a matrix scale and
# a matrix add per stage, which dwarfs the matrix-vector product itself;
# this keeps every operation at matrix-vector cost.
function _minus_im_Hv(Hmats, fs, t, v)
    w = fs[1](t) * (Hmats[1] * v)
    for a in 2:length(Hmats)
        w .+= fs[a](t) .* (Hmats[a] * v)
    end
    return -im .* w
end

"""
Dense RK4 reference for `i dψ/dt = H(t)ψ`, independent of both TDVP and
the MPO code under test. Fourth order in `1/nsub`, so over an O(1)
interval `nsub = 4000` already puts it at ~1e-13 — far below any
tolerance these tests assert — and it is safe as ground truth for
validating a 4th-order integrator.

Unlike comparing against the same method at a large step count, this
cannot land on that method's own roundoff floor: TDVP applied many times
in sequence (as any of these drivers must, at fine step counts) has an
error that *decreases then increases* as steps are refined, because
per-sweep roundoff accumulates once the step is fine enough that the
truncation error drops below it (this is documented behavior, not a bug —
see the Magnus/CFET benchmarks in the README). A "reference" built from
that same method at a very fine step count can therefore be *less*
accurate than a coarser run of the identical method, which silently
invalidates any comparison built on it.
"""
function dense_rk4_reference(Hmats::Vector{<:AbstractMatrix}, fs, v0, t0, t1; nsub = 4000)
    h = (t1 - t0) / nsub
    v = complex(copy(v0))
    for k in 0:(nsub - 1)
        t = t0 + k * h
        k1 = _minus_im_Hv(Hmats, fs, t, v)
        k2 = _minus_im_Hv(Hmats, fs, t + h / 2, v .+ (h / 2) .* k1)
        k3 = _minus_im_Hv(Hmats, fs, t + h / 2, v .+ (h / 2) .* k2)
        k4 = _minus_im_Hv(Hmats, fs, t + h, v .+ h .* k3)
        v = v .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return v
end

"""
Reference time-ordered evolution of `v0` under `H(t) = Σₐ fₐ(t) Hₐ`.

This delegates to [`dense_rk4_reference`](@ref). It previously used a
product of midpoint-frozen matrix exponentials, which was both the
slowest thing in the suite — a full `d × d` exponential *per substep*,
where only the action on one vector is ever needed — and the least
accurate reference available, being second order and so bottoming out
around `1e-8`. RK4 on the vector is asymptotically cheaper per step and
converges four times faster, so it is strictly better on both counts.
"""
function reference_evolution(Hmats, drivings, v0, t0, t1; nsub = 4000)
    return dense_rk4_reference(collect(Hmats), drivings, v0, t0, t1; nsub)
end

"""Infidelity 1 - |⟨a|b⟩| between two normalized dense vectors."""
function infidelity(a, b)
    return 1 - abs(dot(a, b)) / (norm(a) * norm(b))
end
