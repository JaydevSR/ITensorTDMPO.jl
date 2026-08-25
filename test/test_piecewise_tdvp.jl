using ITensorMPS
using ITensors
using ITensorTDMPO
using LinearAlgebra: norm
using Test

# Transverse-field Ising pieces: a static ZZ coupling plus a ramped
# transverse field, which is the archetypal adiabatic ramp setup.
function ising_zz(sites)
    os = OpSum()
    for j in 1:(length(sites) - 1)
        os += -1.0, "Sz", j, "Sz", j + 1
    end
    return MPO(os, sites)
end

function transverse_field(sites, h)
    os = OpSum()
    for j in 1:length(sites)
        os += -h, "Sx", j
    end
    return MPO(os, sites)
end

@testset "DrivenHamiltonian" begin
    sites = siteinds("S=1/2", 4)
    H0 = ising_zz(sites)
    ramp = Ramp(LinearRamp(), 0.0, 1.0, 0.0, 2.0)
    drive = DrivenHamiltonian(H0, t -> transverse_field(sites, ramp(t)))

    ψ = random_mps(sites; linkdims = 4)
    # At t = 0 the drive contributes nothing, so H(0) matches H0.
    @test inner(ψ', drive(0.0), ψ) ≈ inner(ψ', H0, ψ) atol = 1.0e-10
    # At t = 1 the field is at full strength.
    expected = inner(ψ', H0, ψ) + inner(ψ', transverse_field(sites, 2.0), ψ)
    @test inner(ψ', drive(1.0), ψ) ≈ expected atol = 1.0e-10

    # A purely time-dependent Hamiltonian.
    drive_only = DrivenHamiltonian(nothing, t -> transverse_field(sites, ramp(t)))
    @test inner(ψ', drive_only(1.0), ψ) ≈ inner(ψ', transverse_field(sites, 2.0), ψ) atol =
        1.0e-10
end

@testset "piecewise_constant_tdvp matches static tdvp for constant H" begin
    sites = siteinds("S=1/2", 6)
    H0 = ising_zz(sites)
    ψ0 = random_mps(ComplexF64, sites; linkdims = 4)

    # A "time-dependent" part that is actually constant: the piecewise
    # driver must then agree with a plain TDVP evolution.
    Ht = t -> transverse_field(sites, 1.0)
    H_total = +(H0, transverse_field(sites, 1.0); alg = "directsum")

    T = 0.5
    ψ_ref = tdvp(H_total, -im * T, ψ0; nsteps = 10, cutoff = 1.0e-12, maxdim = 64)
    ψ_driver = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, T; nsteps = 10,
        tdvp_kwargs = (; cutoff = 1.0e-12, maxdim = 64)
    )
    @test abs(inner(ψ_ref, ψ_driver)) ≈ 1.0 atol = 1.0e-8
end

@testset "piecewise_constant_tdvp converges under refinement" begin
    sites = siteinds("S=1/2", 6)
    H0 = ising_zz(sites)
    ramp = Ramp(SmoothstepRamp(), 0.0, 1.0, 0.0, 2.0)
    Ht = t -> transverse_field(sites, ramp(t))
    ψ0 = MPS(ComplexF64, sites, "Up")

    tdvp_kwargs = (; cutoff = 1.0e-12, maxdim = 64)
    ψ_coarse = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 10, tdvp_kwargs
    )
    ψ_fine = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 40, tdvp_kwargs
    )
    ψ_finer = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 160, tdvp_kwargs
    )

    # Refining the piecewise-constant grid must bring successive results
    # closer together, i.e. the discretization is actually converging.
    err_coarse = 1 - abs(inner(ψ_coarse, ψ_finer))
    err_fine = 1 - abs(inner(ψ_fine, ψ_finer))
    @test err_fine < err_coarse
    @test err_fine < 1.0e-4

    # Real-time evolution preserves the norm.
    @test norm(ψ_finer) ≈ 1.0 atol = 1.0e-8
end

@testset "schedule and step_observer" begin
    sites = siteinds("S=1/2", 6)
    H0 = ising_zz(sites)
    ramp = Ramp(LinearRamp(), 0.0, 1.0, 0.0, 1.0)
    Ht = t -> transverse_field(sites, ramp(t))
    ψ0 = MPS(ComplexF64, sites, "Up")

    # Reference evolution with plain 2-site TDVP throughout.
    ψ_ref = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 20,
        tdvp_kwargs = (; cutoff = 1.0e-12, maxdim = 64)
    )

    # Switch from 2-site to 1-site partway through, using a global Krylov
    # expansion to keep the 1-site stage from being bond-dimension frozen.
    seen_times = Float64[]
    schedule = function (t, ψ)
        push!(seen_times, t)
        return t < 0.5 ? TDVPStepSpec(; nsite = 2) :
            TDVPStepSpec(;
            nsite = 1, expand_krylov = true,
            expand_kwargs = (; krylovdim = 2, cutoff = 1.0e-8)
        )
    end

    steps = Int[]
    observed = ComplexF64[]
    observer = function (; step, t_start, t_stop, state)
        push!(steps, step)
        push!(observed, inner(state', H0, state))
        return nothing
    end

    ψ_mixed = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 20, schedule,
        step_observer! = observer,
        tdvp_kwargs = (; cutoff = 1.0e-12, maxdim = 64)
    )

    @test steps == collect(1:20)
    @test length(observed) == 20
    @test all(isfinite, real.(observed))
    @test length(seen_times) == 20
    # Midpoint evaluation by default.
    @test seen_times[1] ≈ 0.025
    # The mixed 1-site/2-site evolution tracks the 2-site reference.
    @test abs(inner(ψ_ref, ψ_mixed)) ≈ 1.0 atol = 1.0e-5
end

@testset "eval_at and time grid handling" begin
    sites = siteinds("S=1/2", 4)
    H0 = ising_zz(sites)
    Ht = t -> transverse_field(sites, 0.0)
    ψ0 = MPS(ComplexF64, sites, "Up")

    start_times = Float64[]
    schedule = function (t, ψ)
        push!(start_times, t)
        return TDVPStepSpec(; nsite = 2)
    end
    piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 4, eval_at = :start, schedule
    )
    @test start_times ≈ [0.0, 0.25, 0.5, 0.75]

    @test_throws ErrorException piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; nsteps = 2, eval_at = :end
    )

    # `dt` and `nsteps` must be consistent when both are given.
    @test_throws ErrorException piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; dt = 0.3, nsteps = 4
    )
    # A `dt` that does not evenly divide the interval is an error.
    @test_throws ErrorException piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 1.0; dt = 0.3
    )

    # Passing an explicit, non-uniform time grid.
    grid_times = Float64[]
    grid_schedule = function (t, ψ)
        push!(grid_times, t)
        return TDVPStepSpec(; nsite = 2)
    end
    piecewise_constant_tdvp(
        H0, Ht, ψ0, [0.0, 0.1, 0.5, 1.0]; schedule = grid_schedule
    )
    @test grid_times ≈ [0.05, 0.3, 0.75]
end

@testset "imaginary time evolution lowers the energy" begin
    sites = siteinds("S=1/2", 6)
    H0 = ising_zz(sites)
    Ht = t -> transverse_field(sites, 1.0)
    H_total = +(H0, transverse_field(sites, 1.0); alg = "directsum")
    ψ0 = random_mps(ComplexF64, sites; linkdims = 4)

    energy_before = real(inner(ψ0', H_total, ψ0)) / norm(ψ0)^2
    ψ = piecewise_constant_tdvp(
        H0, Ht, ψ0, 0.0, 2.0; nsteps = 20, generator_prefactor = -1,
        tdvp_kwargs = (; cutoff = 1.0e-12, maxdim = 64, normalize = true)
    )
    energy_after = real(inner(ψ', H_total, ψ)) / norm(ψ)^2
    @test energy_after < energy_before
end
