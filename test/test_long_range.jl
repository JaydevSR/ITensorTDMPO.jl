using ITensorMPS
using ITensors
using TDVPlus
using LinearAlgebra
using Test

isdefined(@__MODULE__, :dense_matrix) || include("dense_reference.jl")

# Nothing in the driving-channel machinery assumes short-range couplings:
# `OpSum` builds an MPO for any two-index term regardless of range, and
# `apply`/`commutator`/`sum` operate on the resulting MPO without regard
# to how it was constructed. This model carries a long-range,
# exponentially-decaying ZZ coupling (as in the paper's own long-range
# benchmarks) alongside a driven transverse field, verified end-to-end
# against a dense reference exactly as the short-range tests are.
function long_range_model(n, α)
    sites = siteinds("S=1/2", n)
    zz = OpSum()
    for i in 1:n, j in (i + 1):n
        zz += exp(-α * (j - i)), "Sz", i, "Sz", j
    end
    x = OpSum()
    for j in 1:n
        x += -1.0, "Sx", j
    end
    return sites, MPO(zz, sites), MPO(x, sites)
end

@testset "long-range MPO construction" begin
    sites, Hzz, Hx = long_range_model(6, 0.7)
    # A genuinely long-range MPO has a larger bond dimension than a
    # nearest-neighbor one, confirming the model actually connects
    # non-adjacent sites rather than silently truncating them.
    @test maxlinkdim(Hzz) > 2

    # Matches an independent dense construction of the same couplings.
    M = dense_matrix(Hzz, sites)
    Mref = zeros(ComplexF64, 2^6, 2^6)
    Sz = [0.5 0; 0 -0.5]
    for i in 1:6, j in (i + 1):6
        op = fill(Matrix{ComplexF64}(I, 2, 2), 6)
        op[i] = Sz
        op[j] = Sz
        Mref += exp(-0.7 * (j - i)) * kron(reverse(op)...)
    end
    @test M ≈ Mref rtol = 1.0e-8
end

@testset "Dyson and Magnus with long-range interactions" begin
    sites, Hzz, Hx = long_range_model(6, 0.7)
    fz = t -> 1.0
    fx = t -> sin(3t) + 0.6
    ch = DrivingChannels([(fz, Hzz), (fx, Hx)])
    ψ0 = MPS(ComplexF64, sites, "Up")

    Mzz, Mx = dense_matrix(Hzz, sites), dense_matrix(Hx, sites)
    v0 = dense_vector(ψ0, sites)
    T = 0.4
    function rk4(nsub)
        h = T / nsub
        v = complex(copy(v0))
        Ht(t) = fz(t) * Mzz + fx(t) * Mx
        for k in 0:(nsub - 1)
            t = k * h
            k1 = -im * (Ht(t) * v)
            k2 = -im * (Ht(t + h / 2) * (v .+ (h / 2) .* k1))
            k3 = -im * (Ht(t + h / 2) * (v .+ (h / 2) .* k2))
            k4 = -im * (Ht(t + h) * (v .+ h .* k3))
            v = v .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
        return v
    end
    vex = rk4(20_000)
    td(ψ) = begin
        v = dense_vector(ψ, sites)
        ov = abs(dot(v, vex)) / (norm(v) * norm(vex))
        sqrt(max(0.0, 1 - min(1.0, ov)^2))
    end

    EX = (; cutoff = 1.0e-14, maxdim = 256)
    for alg in ("magnus", "dyson", "cfet", "piecewise_constant")
        order = alg == "piecewise_constant" ? nothing : (alg == "cfet" ? 4 : 2)
        kw = (; alg, nsteps = 12, cutoff = EX.cutoff, maxdim = EX.maxdim,
            operator_cutoff = EX.cutoff, operator_maxdim = EX.maxdim)
        ψ = isnothing(order) ? time_evolve(ch, ψ0, 0.0, T; kw...) :
            time_evolve(ch, ψ0, 0.0, T; order, kw...)
        @test td(ψ) < 1.0e-3
    end
end
