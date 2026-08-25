using ITensorMPS
using ITensors
using ITensorTDMPO
using LinearAlgebra
using Test

function obs_model(n = 6)
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

@testset "entanglement_entropy" begin
    sites, _, _ = obs_model(6)

    # A product state has zero entanglement across every bond.
    prod = MPS(ComplexF64, sites, "Up")
    @test entanglement_entropy(prod) ≈ 0.0 atol = 1.0e-10
    @test all(≈(0.0; atol = 1.0e-10), entanglement_profile(prod))

    # A Bell pair on two sites has entropy log 2 (1 bit).
    s2 = siteinds("S=1/2", 2)
    bell = MPS([1 / sqrt(2), 0.0, 0.0, 1 / sqrt(2)], s2)
    @test entanglement_entropy(bell, 1) ≈ log(2) rtol = 1.0e-8
    @test entanglement_entropy(bell, 1; base = 2) ≈ 1.0 rtol = 1.0e-8

    # Unnormalized input gives the same answer: Schmidt values are
    # renormalized internally.
    @test entanglement_entropy(3.7 * bell, 1) ≈ log(2) rtol = 1.0e-8

    @test length(entanglement_profile(prod)) == length(sites) - 1
    @test_throws ArgumentError entanglement_entropy(prod, 0)
    @test_throws ArgumentError entanglement_entropy(prod, length(sites))
end

@testset "energy and variance" begin
    sites, Hzz, Hx = obs_model(6)
    ch = DrivingChannels([(1.0, Hzz), (0.5, Hx)])
    ψ = random_mps(ComplexF64, sites; linkdims = 4)

    # Energy matches a direct expectation value.
    H = ch(0.0)
    @test instantaneous_energy(ch, 0.0, ψ) ≈ real(inner(ψ', H, ψ)) / real(inner(ψ, ψ)) rtol =
        1.0e-8

    # Both are invariant under rescaling the state.
    @test instantaneous_energy(ch, 0.0, 2.5 * ψ) ≈ instantaneous_energy(ch, 0.0, ψ) rtol = 1.0e-8
    @test energy_variance(ch, 0.0, 2.5 * ψ) ≈ energy_variance(ch, 0.0, ψ) rtol = 1.0e-6

    # Variance is non-negative, and vanishes on an eigenstate.
    @test energy_variance(ch, 0.0, ψ) > 0
    zz_only = DrivingChannels([(1.0, Hzz)])
    up = MPS(ComplexF64, sites, "Up")     # eigenstate of the ZZ term
    @test energy_variance(zz_only, 0.0, up) ≈ 0.0 atol = 1.0e-8
end

@testset "instantaneous spectrum and gap" begin
    sites, Hzz, Hx = obs_model(6)
    ch = DrivingChannels([(1.0, Hzz), (0.6, Hx)])
    ψ_guess = random_mps(ComplexF64, sites; linkdims = 8)
    dk = (; nsweeps = 12, maxdim = 64, cutoff = 1.0e-12)

    energies, states = instantaneous_spectrum(ch, 0.0, ψ_guess; nlevels = 2, dk...)
    @test length(energies) == 2 && length(states) == 2
    @test energies[1] <= energies[2]

    # The ground energy must match a direct DMRG solve on H(0).
    E_ref, _ = dmrg(ch(0.0), random_mps(ComplexF64, sites; linkdims = 8); dk..., outputlevel = 0)
    @test energies[1] ≈ E_ref rtol = 1.0e-5

    gap = instantaneous_gap(ch, 0.0, ψ_guess; dk...)
    @test gap ≈ energies[2] - energies[1] rtol = 1.0e-4
    @test gap > 0

    # The ground state has (near) zero energy variance.
    @test energy_variance(ch, 0.0, states[1]) < 1.0e-4
end

@testset "adiabatic_report" begin
    sites, Hzz, Hx = obs_model(6)
    ch = DrivingChannels([(1.0, Hzz), (0.6, Hx)])
    ψ = random_mps(ComplexF64, sites; linkdims = 6)

    cheap = adiabatic_report(ch, 0.2, ψ)
    @test cheap.energy ≈ instantaneous_energy(ch, 0.2, ψ) rtol = 1.0e-8
    @test cheap.variance ≈ energy_variance(ch, 0.2, ψ) rtol = 1.0e-6
    # Without `gap = true` the DMRG-based fields are absent, not zero.
    @test ismissing(cheap.gap)
    @test ismissing(cheap.excess)
    @test ismissing(cheap.fidelity)

    full = adiabatic_report(
        ch, 0.2, ψ; gap = true, nsweeps = 12, maxdim = 64, cutoff = 1.0e-12
    )
    @test full.gap > 0
    @test full.excess >= -1.0e-6            # ⟨H⟩ cannot be below E₀
    @test 0 <= full.fidelity <= 1 + 1.0e-8
end

@testset "EvolutionObserver" begin
    sites, Hzz, Hx = obs_model(6)
    ch = DrivingChannels([(1.0, Hzz), (t -> sin(4t) + 0.5, Hx)])
    ψ0 = MPS(ComplexF64, sites, "Up")

    obs = EvolutionObserver(
        :chi => maxlinkdim,
        :entropy => ψ -> entanglement_entropy(ψ),
        :energy => (ψ, t) -> instantaneous_energy(ch, t, ψ),
    )
    @test length(obs) == 0

    observe!(obs, ψ0, 0.0)                    # seed with the initial state
    @test length(obs) == 1

    ψ = time_evolve(ch, ψ0, 0.0, 0.4; nsteps = 4, step_observer! = obs)
    @test length(obs) == 5

    r = results(obs)
    @test r.time ≈ [0.0, 0.1, 0.2, 0.3, 0.4]
    @test r.step == [1, 1, 2, 3, 4]
    @test length(r.chi) == 5
    # Element types are narrowed, not left as Any.
    @test r.entropy isa Vector{Float64}
    @test r.chi isa Vector{Int}
    # The initial product state has zero entropy; evolution creates some.
    @test r.entropy[1] ≈ 0.0 atol = 1.0e-10
    @test r.entropy[end] > 1.0e-6
    # The two-argument measurement really saw the time.
    @test r.energy[1] ≈ instantaneous_energy(ch, 0.0, ψ0) rtol = 1.0e-8

    # Reusable after emptying.
    empty!(obs)
    @test length(obs) == 0
    @test isempty(results(obs).time)

    # `every = k` subsamples, so expensive diagnostics can run sparsely.
    sparse_obs = EvolutionObserver(:chi => maxlinkdim; every = 2)
    time_evolve(ch, ψ0, 0.0, 0.6; nsteps = 6, step_observer! = sparse_obs)
    @test length(sparse_obs) == 3
    @test results(sparse_obs).step == [2, 4, 6]
    # An explicit observe! records regardless of `every`.
    observe!(sparse_obs, ψ0, 99.0)
    @test length(sparse_obs) == 4

    # Malformed observers are rejected.
    @test_throws ArgumentError EvolutionObserver()
    @test_throws ArgumentError EvolutionObserver(:a => maxlinkdim; every = 0)
    @test_throws ArgumentError EvolutionObserver(:a => maxlinkdim, :a => maxlinkdim)
    @test_throws ArgumentError EvolutionObserver(:a => "not callable")
end
