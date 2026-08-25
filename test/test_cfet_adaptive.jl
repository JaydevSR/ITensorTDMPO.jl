using ITensorMPS
using ITensors
using TDVPlus
using LinearAlgebra
using Random
using Test

isdefined(@__MODULE__, :dense_matrix) || include("dense_reference.jl")

function ca_model(n = 6)
    sites = siteinds("S=1/2", n)
    zz = OpSum()
    for j in 1:(n - 1)
        zz += -1.0, "Sz", j, "Sz", j + 1
    end
    x = OpSum()
    for j in 1:n
        x += -1.0, "Sx", j
    end
    return sites, MPO(zz, sites), MPO(x, sites)
end

const CA_EXACT = (; cutoff = 1.0e-14, maxdim = 256)

# `dense_rk4_reference` now lives in `dense_reference.jl`, shared with
# the Dyson/Magnus tests.

"""Trace distance between an MPS `ψ` and a dense reference vector `vex`."""
function trace_distance_dense(ψ::MPS, vex, sites)
    v = dense_vector(ψ, sites)
    ov = abs(dot(v, vex)) / (norm(v) * norm(vex))
    return sqrt(max(0.0, 1 - min(1.0, ov)^2))
end

@testset "trace_distance" begin
    # `trace_distance` is `sqrt(1 - ov^2)`, which has an infinite
    # derivative at `ov = 1`: for two states that are exactly identical
    # (up to scale) the *true* overlap is 1, but the *computed* overlap
    # carries a few ULPs of floating-point noise from the underlying
    # tensor contraction (measured: `1 - ov` at the 1e-16–1e-15 level
    # across 20 random draws). The square root amplifies that into an
    # O(1e-8) output whenever it rounds to just below 1 — up to 3.7e-8
    # measured — so `atol` here must be well above machine epsilon, not
    # at it.
    sites, _, _ = ca_model(4)
    a = random_mps(ComplexF64, sites; linkdims = 4)
    @test trace_distance(a, a) ≈ 0.0 atol = 1.0e-6
    # Insensitive to normalization.
    @test trace_distance(a, 3.2 * a) ≈ 0.0 atol = 1.0e-6
    b = random_mps(ComplexF64, sites; linkdims = 4)
    @test 0 <= trace_distance(a, b) <= 1 + 1.0e-8
    @test trace_distance(a, b) ≈ trace_distance(b, a) rtol = 1.0e-8
end

@testset "cfet_exponents" begin
    sites, Hzz, Hx = ca_model(5)
    f = t -> 2.0        # constant driving makes the weights checkable
    g = t -> 3.0
    ch = DrivingChannels([(f, Hzz), (g, Hx)])
    Mzz, Mx = dense_matrix(Hzz, sites), dense_matrix(Hx, sites)

    # Order 2 is the midpoint rule: a single exponent equal to H(midpoint).
    W2 = cfet_exponents(ch, 0.0, 0.4; order = 2)
    @test length(W2) == 1
    @test dense_matrix(only(W2), sites) ≈ 2.0 * Mzz + 3.0 * Mx rtol = 1.0e-8

    # Order 4 gives two exponents whose weights sum to the full step:
    # with constant driving each Wⱼ = (a₁+a₂) H = H/2.
    W4 = cfet_exponents(ch, 0.0, 0.4; order = 4)
    @test length(W4) == 2
    Htot = 2.0 * Mzz + 3.0 * Mx
    for W in W4
        @test dense_matrix(W, sites) ≈ Htot / 2 rtol = 1.0e-8
    end
    # ...and together they reproduce the whole step.
    @test dense_matrix(W4[1], sites) + dense_matrix(W4[2], sites) ≈ Htot rtol = 1.0e-8

    @test_throws ArgumentError cfet_exponents(ch, 0.0, 0.4; order = 3)
end

@testset "cfet reproduces exact evolution for constant H" begin
    # With a time-independent Hamiltonian every scheme must agree with
    # exp(-iTH) applied directly.
    sites, Hzz, Hx = ca_model(6)
    ch = DrivingChannels([(1.0, Hzz), (0.7, Hx)])
    ψ0 = MPS(ComplexF64, sites, "Up")
    T = 0.3
    ref = tdvp(ch(0.0), -im * T, ψ0; nsteps = 40, CA_EXACT...)

    for (alg, order) in (("cfet", 4), ("cfet", 2), ("magnus", 2))
        ψ = time_evolve(
            ch, ψ0, 0.0, T; alg, order, nsteps = 8,
            cutoff = CA_EXACT.cutoff, maxdim = CA_EXACT.maxdim,
            operator_cutoff = 1.0e-14, operator_maxdim = 256
        )
        @test trace_distance(ψ, ref) < 1.0e-6
    end
end

@testset "cfet is fourth order on a driven model" begin
    # An independent dense RK4 reference, not the same method at a large
    # step count: CFET/TDVP's own roundoff floor (see `dense_rk4_reference`
    # docstring) means a self-referential "exact" run can be *less*
    # accurate than the runs it is meant to validate, as happened here
    # with a naive `nsteps = 256` self-reference.
    sites, Hzz, Hx = ca_model(6)
    fz = t -> sin(2pi * t)
    fx = t -> cos(2pi * t)
    ch = DrivingChannels([(fz, Hzz), (fx, Hx)])
    ψ0 = MPS(ComplexF64, sites, "Up")
    T = 0.5
    kw = (;
        cutoff = 1.0e-14, maxdim = 256,
        operator_cutoff = 1.0e-14, operator_maxdim = 256,
    )
    Mzz, Mx = dense_matrix(Hzz, sites), dense_matrix(Hx, sites)
    v0 = dense_vector(ψ0, sites)
    vex = dense_rk4_reference([Mzz, Mx], [fz, fx], v0, 0.0, T)
    err(ψ) = trace_distance_dense(ψ, vex, sites)

    e1 = err(time_evolve(ch, ψ0, 0.0, T; alg = "cfet", order = 4, nsteps = 4, kw...))
    e2 = err(time_evolve(ch, ψ0, 0.0, T; alg = "cfet", order = 4, nsteps = 8, kw...))
    # Halving the step should cut the error by ~2⁴ = 16; allow slack. Both
    # step counts are well inside the convergent regime (the roundoff
    # floor for this model only appears past nsteps ≈ 16–32).
    @test e2 < e1 / 8

    # Order 2 must be genuinely worse than order 4 at the same step count.
    e_o2 = err(time_evolve(ch, ψ0, 0.0, T; alg = "cfet", order = 2, nsteps = 8, kw...))
    @test e_o2 > e2
end

@testset "adaptive stepping" begin
    sites, Hzz, Hx = ca_model(6)
    fz = t -> sin(2pi * t)
    fx = t -> cos(2pi * t)
    ch = DrivingChannels([(fz, Hzz), (fx, Hx)])
    ψ0 = MPS(ComplexF64, sites, "Up")
    T = 0.5
    kw = (;
        cutoff = 1.0e-14, maxdim = 256,
        operator_cutoff = 1.0e-14, operator_maxdim = 256,
    )
    # Independent dense reference; see the note in the previous testset on
    # why a self-referential high-step-count run is not safe here.
    Mzz, Mx = dense_matrix(Hzz, sites), dense_matrix(Hx, sites)
    v0 = dense_vector(ψ0, sites)
    vex = dense_rk4_reference([Mzz, Mx], [fz, fx], v0, 0.0, T)
    err(ψ) = trace_distance_dense(ψ, vex, sites)

    ψ, hist = adaptive_time_evolve(
        ch, ψ0, 0.0, T; alg = "cfet", order = 4, tol = 1.0e-6, kw...
    )
    @test hist.times[1] ≈ 0.0
    @test hist.times[end] ≈ T rtol = 1.0e-10
    @test sum(hist.dts) ≈ T rtol = 1.0e-10
    @test length(hist.dts) == length(hist.errors) == length(hist.times) - 1
    @test all(<=(1.0e-6 + 1.0e-12), hist.errors)
    @test err(ψ) < 1.0e-5

    # A tighter tolerance must take more steps and land closer — but only
    # within the regime where step-doubling error estimation is still
    # meaningful. Below `tol ≈ 1e-7` for this problem, the coarse and fine
    # sub-steps become indistinguishable at TDVP's own per-application
    # roundoff floor (the internal error estimate collapses to exactly
    # `0.0`); the controller then keeps refining against noise, and the
    # *true* error gets worse, not better, as more steps accumulate more
    # roundoff. `1e-5` and `1e-6` are both comfortably above that floor
    # (confirmed by comparison against the independent dense reference).
    ψ_t, hist_t = adaptive_time_evolve(
        ch, ψ0, 0.0, T; alg = "cfet", order = 4, tol = 1.0e-5, kw...
    )
    @test length(hist.dts) >= length(hist_t.dts)
    @test err(ψ) <= err(ψ_t) * 1.5

    # Reachable through `time_evolve(...; adaptive = true)`.
    ψ_te = time_evolve(
        ch, ψ0, 0.0, T; alg = "cfet", order = 4, adaptive = true, tol = 1.0e-7, kw...
    )
    @test trace_distance(ψ_te, ψ) < 1.0e-6

    # A zero-length interval is a no-op, not an error.
    ψ_z, hist_z = adaptive_time_evolve(ch, ψ0, 0.3, 0.3; alg = "cfet", order = 4)
    @test isempty(hist_z.dts)
    @test trace_distance(ψ_z, ψ0) ≈ 0.0 atol = 1.0e-10

    # An unreachable tolerance fails loudly rather than stalling.
    @test_throws ErrorException adaptive_time_evolve(
        ch, ψ0, 0.0, T; alg = "cfet", order = 4, tol = 1.0e-16, dt_min = 1.0e-3, kw...
    )
end

@testset "imaginary time lowers the energy" begin
    # Seeded for reproducibility: the final assertion compares the imaginary-
    # time-projected energy against the true ground energy at a fixed `T`,
    # and how close that gets depends on this state's overlap with the
    # ground and first excited state. Left unseeded, the test's pass/fail
    # depended on which random state was drawn (see the margin check below).
    Random.seed!(1234)
    sites, Hzz, Hx = ca_model(6)
    ch = DrivingChannels([(1.0, Hzz), (0.8, Hx)])
    ψ0 = random_mps(ComplexF64, sites; linkdims = 4)
    E0 = instantaneous_energy(ch, 0.0, ψ0)

    for alg in ("magnus", "cfet", "piecewise_constant")
        order = alg == "piecewise_constant" ? nothing : (alg == "cfet" ? 4 : 2)
        kw = (;
            alg, nsteps = 20, generator_prefactor = -1, normalize = true,
            cutoff = 1.0e-12, maxdim = 64,
        )
        ψ = isnothing(order) ?
            time_evolve(ch, ψ0, 0.0, 2.0; kw...) :
            time_evolve(ch, ψ0, 0.0, 2.0; order, kw...)
        E = instantaneous_energy(ch, 0.0, ψ)
        @test E < E0                      # imaginary time must lower it
        @test norm(ψ) ≈ 1.0 atol = 1.0e-8  # normalize kept it finite
    end

    # All three should approach the same ground energy.
    E_dmrg, _ = dmrg(
        ch(0.0), random_mps(ComplexF64, sites; linkdims = 8);
        nsweeps = 12, maxdim = 64, cutoff = 1.0e-12, outputlevel = 0
    )
    # `T = 10` (dt = 0.1) at this model's gap (0.382) leaves a comfortable
    # margin below `rtol`: measured worst case over 10 independent random
    # initial states was 2.7e-4, well under the 1e-3 threshold. `T = 6`
    # (used previously) sits right at the edge of the convergence curve
    # for this gap — errors ranged 2.3e-4 to 9.3e-3 across seeds, i.e. the
    # test could pass or fail depending on which random state was drawn.
    ψ_it = time_evolve(
        ch, ψ0, 0.0, 10.0; alg = "cfet", order = 4, nsteps = 100,
        generator_prefactor = -1, normalize = true, cutoff = 1.0e-12, maxdim = 64
    )
    @test instantaneous_energy(ch, 0.0, ψ_it) ≈ E_dmrg rtol = 1.0e-3
end
