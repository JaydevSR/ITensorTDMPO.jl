using ITensorMPSExtended
using Test

@testset "ramp shapes" begin
    shapes = [
        LinearRamp(),
        SmoothstepRamp(),
        SmootherstepRamp(),
        SineRamp(),
        SineSquaredRamp(),
        PowerLawRamp(2.0),
        ExponentialRamp(3.0),
        ExponentialRamp(0.0),
    ]
    for shape in shapes
        @test shape(0.0) ≈ 0.0 atol = 1.0e-12
        @test shape(1.0) ≈ 1.0 atol = 1.0e-12
        # Monotonic and within bounds on the interior.
        τs = range(0, 1; length = 51)
        ss = shape.(τs)
        @test all(0 .<= ss .<= 1)
        @test all(diff(ss) .>= -1.0e-12)
        # Clamped outside [0, 1].
        @test shape(-1.0) ≈ 0.0 atol = 1.0e-12
        @test shape(2.0) ≈ 1.0 atol = 1.0e-12
    end

    # Symmetric shapes hit the halfway point at τ = 1/2.
    for shape in [
            LinearRamp(), SmoothstepRamp(), SmootherstepRamp(), SineRamp(),
            SineSquaredRamp(),
        ]
        @test shape(0.5) ≈ 0.5 atol = 1.0e-12
    end

    # sin²(πτ/2) == (1 - cos(πτ))/2 by the half-angle identity.
    for τ in range(0, 1; length = 21)
        @test SineSquaredRamp()(τ) ≈ SineRamp()(τ) atol = 1.0e-12
    end

    @test ExponentialRamp(0.0)(0.37) ≈ LinearRamp()(0.37)
    @test PowerLawRamp(1.0)(0.37) ≈ LinearRamp()(0.37)
end

@testset "Ramp" begin
    ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 1.0)
    @test ramp(0.0) ≈ 0.0
    @test ramp(5.0) ≈ 0.5
    @test ramp(10.0) ≈ 1.0
    @test ramp(-5.0) ≈ 0.0
    @test ramp(15.0) ≈ 1.0

    # Descending ramp over a shifted window.
    descending = Ramp(LinearRamp(), 2.0, 6.0, 4.0, -4.0)
    @test descending(2.0) ≈ 4.0
    @test descending(4.0) ≈ 0.0
    @test descending(6.0) ≈ -4.0
    @test descending(100.0) ≈ -4.0

    # Integer arguments are promoted, not an error.
    @test Ramp(LinearRamp(), 0, 4, 0, 1)(2) ≈ 0.5
end
