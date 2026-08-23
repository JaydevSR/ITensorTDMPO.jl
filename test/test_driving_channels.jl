using ITensorMPS
using ITensors
using ITensorMPSExtended
using LinearAlgebra
using Test

isdefined(@__MODULE__, :dense_matrix) || include("dense_reference.jl")

@testset "DrivingChannels construction" begin
    sites = siteinds("S=1/2", 4)
    os1 = OpSum()
    for j in 1:3
        os1 += -1.0, "Sz", j, "Sz", j + 1
    end
    os2 = OpSum()
    for j in 1:4
        os2 += -1.0, "Sx", j
    end
    H1, H2 = MPO(os1, sites), MPO(os2, sites)
    f = t -> 1.0
    g = t -> sin(t)

    ch = DrivingChannels(H1 => f, H2 => g)
    @test nchannels(ch) == 2
    @test ITensorMPS.siteinds(ch) == sites
    @test ch.operators[1] === H1
    @test ch.drivings[2] === g

    # Vector-of-pairs and explicit-vector constructors agree.
    ch2 = DrivingChannels([H1 => f, H2 => g])
    ch3 = DrivingChannels(MPO[H1, H2], [f, g])
    @test nchannels(ch2) == nchannels(ch3) == 2

    # Mismatched lengths and empty input are rejected.
    @test_throws ArgumentError DrivingChannels(MPO[H1, H2], [f])
    @test_throws ArgumentError DrivingChannels(MPO[], [])

    # Channels on different site sets are rejected.
    other = siteinds("S=1/2", 4)
    os3 = OpSum()
    for j in 1:4
        os3 += -1.0, "Sx", j
    end
    @test_throws ArgumentError DrivingChannels(MPO[H1, MPO(os3, other)], [f, g])

    # The identity MPO acts as the identity.
    ψ = random_mps(ComplexF64, sites; linkdims = 3)
    @test abs(inner(ψ, apply(identity_mpo(ch), ψ))) ≈ norm(ψ)^2 rtol = 1.0e-8
end

@testset "H(t) assembly and dense conversion" begin
    sites = siteinds("S=1/2", 5)
    os1 = OpSum()
    for j in 1:4
        os1 += -1.0, "Sz", j, "Sz", j + 1
    end
    os2 = OpSum()
    for j in 1:5
        os2 += -1.0, "Sx", j
    end
    H1, H2 = MPO(os1, sites), MPO(os2, sites)
    f = t -> 0.7
    g = t -> sin(t)
    ch = DrivingChannels(H1 => f, H2 => g)

    # Validate the dense conversion itself: applying the MPO and
    # multiplying by the dense matrix must agree.
    ψ = random_mps(ComplexF64, sites; linkdims = 4)
    v = dense_vector(ψ, sites)
    M1 = dense_matrix(H1, sites)
    @test dense_vector(apply(H1, ψ; cutoff = 1.0e-14), sites) ≈ M1 * v rtol = 1.0e-8

    # H(t) = Σₐ fₐ(t) H^{(a)}.
    t = 0.6
    Ht = ch(t)
    M2 = dense_matrix(H2, sites)
    @test dense_matrix(Ht, sites) ≈ f(t) * M1 + g(t) * M2 rtol = 1.0e-8
end

@testset "commutator" begin
    sites = siteinds("S=1/2", 5)
    os1 = OpSum()
    for j in 1:4
        os1 += -1.0, "Sz", j, "Sz", j + 1
    end
    os2 = OpSum()
    for j in 1:5
        os2 += -1.0, "Sx", j
    end
    H1, H2 = MPO(os1, sites), MPO(os2, sites)
    M1, M2 = dense_matrix(H1, sites), dense_matrix(H2, sites)

    C = commutator(H1, H2)
    @test dense_matrix(C, sites) ≈ M1 * M2 - M2 * M1 rtol = 1.0e-8

    # Antisymmetry, and a self-commutator that vanishes.
    @test dense_matrix(commutator(H2, H1), sites) ≈ -(M1 * M2 - M2 * M1) rtol = 1.0e-8
    @test norm(dense_matrix(commutator(H1, H1), sites)) < 1.0e-8

    # The commutator of two Hermitian operators is anti-Hermitian.
    D = dense_matrix(C, sites)
    @test norm(D + D') < 1.0e-8
end
