using ITensorMPS
using ITensors
using ITensorMPSExtended
using LinearAlgebra
using Test

isdefined(@__MODULE__, :dense_matrix) || include("dense_reference.jl")

# A driven transverse-field Ising chain: a static ZZ coupling plus a
# transverse field with a nontrivial (non-commuting, time-dependent)
# drive. Small enough that dense reference evolution is exact.
function test_model(n = 6)
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

# Exact-for-this-size truncation settings, so that the only error under
# test is the Dyson/Magnus expansion error and not MPS truncation.
const EXACT = (; cutoff = 1.0e-14, maxdim = 256)

@testset "Dyson MPO reduces to the Taylor series for constant driving" begin
    sites, Hzz, Hx = test_model(5)
    # A single channel with f ≡ 1 makes H(t) time independent, so the
    # Dyson series must reduce to the truncated Taylor series of
    # exp(-i Δ H) — this is the time-independent limit of Sec. V.
    ch = DrivingChannels(Hzz => (t -> 1.0))
    Δ = 0.35
    M = dense_matrix(Hzz, sites)

    for order in 1:3
        U = dyson_mpo(ch, 0.0, Δ; order, EXACT...)
        taylor = sum(((-im * Δ)^k / factorial(k)) * M^k for k in 0:order)
        @test dense_matrix(U, sites) ≈ taylor rtol = 1.0e-7
    end

    # Increasing the order approaches the true exponential.
    exact = exp(-im * Δ * Matrix(M))
    errs = [
        norm(dense_matrix(dyson_mpo(ch, 0.0, Δ; order, EXACT...), sites) - exact)
            for order in 1:3
    ]
    @test issorted(errs; rev = true)
end

@testset "Magnus generator for a single channel" begin
    sites, Hzz, _ = test_model(5)
    # With one channel all commutators vanish, so Ω = [f] H exactly at
    # every order and exp(Ω) is the exact propagator.
    f = t -> 1.0 + 0.5 * sin(3t)
    ch = DrivingChannels(Hzz => f)
    t0, t1 = 0.0, 0.4
    M = dense_matrix(Hzz, sites)
    coeff = time_ordered_integral([f], t0, t1)

    for order in 1:3
        Ω = magnus_generator(ch, t0, t1; order, EXACT...)
        @test dense_matrix(Ω, sites) ≈ coeff * M rtol = 1.0e-7
    end
end

@testset "Magnus generator is anti-Hermitian" begin
    sites, Hzz, Hx = test_model(5)
    ch = DrivingChannels(Hzz => (t -> 1.0), Hx => (t -> sin(2t) + 0.4))
    for order in 1:3
        Ω = magnus_generator(ch, 0.1, 0.5; order, EXACT...)
        D = dense_matrix(Ω, sites)
        @test norm(D + D') / norm(D) < 1.0e-7
    end
end

@testset "first-order Dyson and Magnus agree" begin
    # Vanthilt et al. note that the first-order Dyson MPO is equivalent
    # to the first-order Magnus operator exponentiated to first order,
    # i.e. 1 + Ω₁.
    sites, Hzz, Hx = test_model(5)
    ch = DrivingChannels(Hzz => (t -> 1.0), Hx => (t -> cos(t)))
    t0, t1 = 0.0, 0.3
    U1 = dyson_mpo(ch, t0, t1; order = 1, EXACT...)
    Ω1 = magnus_generator(ch, t0, t1; order = 1, EXACT...)
    Id = dense_matrix(identity_mpo(ch), sites)
    @test dense_matrix(U1, sites) ≈ Id + dense_matrix(Ω1, sites) rtol = 1.0e-7
end

@testset "Dyson series against exact evolution" begin
    sites, Hzz, Hx = test_model(6)
    f = t -> 1.0
    g = t -> sin(4t) + 0.5
    ch = DrivingChannels(Hzz => f, Hx => g)
    Hmats = [dense_matrix(Hzz, sites), dense_matrix(Hx, sites)]

    ψ0 = MPS(ComplexF64, sites, "Up")
    v0 = dense_vector(ψ0, sites)
    T = 0.5
    v_exact = reference_evolution(Hmats, [f, g], v0, 0.0, T; nsub = 8000)

    # Higher order gives a smaller error at fixed step count.
    errs = Float64[]
    for order in 1:3
        ψ = dyson_evolve(
            ch, ψ0, 0.0, T; nsteps = 4, order,
            mpo_kwargs = EXACT, apply_kwargs = EXACT
        )
        push!(errs, infidelity(dense_vector(ψ, sites), v_exact))
    end
    @test issorted(errs; rev = true)
    @test errs[3] < 1.0e-5

    # Halving the step size reduces the order-2 error by roughly 2² = 4.
    e_coarse = infidelity(
        dense_vector(
            dyson_evolve(
                ch, ψ0, 0.0, T; nsteps = 4, order = 2,
                mpo_kwargs = EXACT, apply_kwargs = EXACT
            ), sites
        ), v_exact
    )
    e_fine = infidelity(
        dense_vector(
            dyson_evolve(
                ch, ψ0, 0.0, T; nsteps = 8, order = 2,
                mpo_kwargs = EXACT, apply_kwargs = EXACT
            ), sites
        ), v_exact
    )
    @test e_fine < e_coarse / 3
end

@testset "Magnus expansion against exact evolution" begin
    sites, Hzz, Hx = test_model(6)
    f = t -> 1.0
    g = t -> sin(4t) + 0.5
    ch = DrivingChannels(Hzz => f, Hx => g)
    Hmats = [dense_matrix(Hzz, sites), dense_matrix(Hx, sites)]

    ψ0 = MPS(ComplexF64, sites, "Up")
    v0 = dense_vector(ψ0, sites)
    T = 0.5
    v_exact = reference_evolution(Hmats, [f, g], v0, 0.0, T; nsub = 8000)

    tdvp_kwargs = (; cutoff = 1.0e-14, maxdim = 256, nsteps = 4)
    errs = Float64[]
    for order in 1:3
        ψ = magnus_evolve(
            ch, ψ0, 0.0, T; nsteps = 4, order,
            generator_kwargs = EXACT, tdvp_kwargs
        )
        push!(errs, infidelity(dense_vector(ψ, sites), v_exact))
    end
    @test issorted(errs; rev = true)
    @test errs[2] < 1.0e-4

    # Magnus evolution is unitary, so the norm is preserved exactly.
    ψ = magnus_evolve(
        ch, ψ0, 0.0, T; nsteps = 4, order = 2,
        generator_kwargs = EXACT, tdvp_kwargs
    )
    @test norm(ψ) ≈ 1.0 atol = 1.0e-8
end

@testset "Dyson and Magnus beat freezing the Hamiltonian" begin
    # The point of both expansions: with the same number of steps, a
    # rapidly oscillating drive is captured far better than by holding
    # H(t) constant across each step.
    sites, Hzz, Hx = test_model(6)
    f = t -> 1.0
    g = t -> sin(12t)
    ch = DrivingChannels(Hzz => f, Hx => g)
    Hmats = [dense_matrix(Hzz, sites), dense_matrix(Hx, sites)]

    ψ0 = MPS(ComplexF64, sites, "Up")
    v0 = dense_vector(ψ0, sites)
    T = 1.0
    nsteps = 8
    v_exact = reference_evolution(Hmats, [f, g], v0, 0.0, T; nsub = 20000)

    ψ_frozen = piecewise_constant_tdvp(
        Hzz, t -> g(t) * Hx, ψ0, 0.0, T; nsteps,
        tdvp_kwargs = (; cutoff = 1.0e-14, maxdim = 256, nsteps = 4)
    )
    ψ_dyson = dyson_evolve(
        ch, ψ0, 0.0, T; nsteps, order = 3, mpo_kwargs = EXACT, apply_kwargs = EXACT
    )
    ψ_magnus = magnus_evolve(
        ch, ψ0, 0.0, T; nsteps, order = 3, generator_kwargs = EXACT,
        tdvp_kwargs = (; cutoff = 1.0e-14, maxdim = 256, nsteps = 4)
    )

    e_frozen = infidelity(dense_vector(ψ_frozen, sites), v_exact)
    e_dyson = infidelity(dense_vector(ψ_dyson, sites), v_exact)
    e_magnus = infidelity(dense_vector(ψ_magnus, sites), v_exact)

    @test e_dyson < e_frozen
    @test e_magnus < e_frozen
end

@testset "argument validation" begin
    sites, Hzz, Hx = test_model(4)
    ch = DrivingChannels(Hzz => (t -> 1.0), Hx => (t -> 1.0))
    @test_throws ArgumentError dyson_mpo(ch, 0.0, 0.1; order = -1)
    @test_throws ArgumentError magnus_generator(ch, 0.0, 0.1; order = 0)
    @test_throws ArgumentError magnus_generator(ch, 0.0, 0.1; order = 4)

    # Order 0 is the identity.
    U0 = dyson_mpo(ch, 0.0, 0.1; order = 0, EXACT...)
    @test dense_matrix(U0, sites) ≈ dense_matrix(identity_mpo(ch), sites) rtol = 1.0e-10
end

@testset "step observers" begin
    sites, Hzz, Hx = test_model(5)
    ch = DrivingChannels(Hzz => (t -> 1.0), Hx => (t -> cos(t)))
    ψ0 = MPS(ComplexF64, sites, "Up")

    for evolve in (dyson_evolve, magnus_evolve)
        steps = Int[]
        stops = Float64[]
        observer = function (; step, t_start, t_stop, state)
            push!(steps, step)
            push!(stops, t_stop)
            return nothing
        end
        evolve(ch, ψ0, 0.0, 0.4; nsteps = 4, order = 1, step_observer! = observer)
        @test steps == collect(1:4)
        @test stops ≈ [0.1, 0.2, 0.3, 0.4]
    end
end
