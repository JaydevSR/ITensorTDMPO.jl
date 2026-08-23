using ITensorMPSExtended
using Test

@testset "cumulative_integral" begin
    # ∫₀ˣ cos = sin
    grid = range(0, 2π; length = 401)
    h = step(grid)
    F = cumulative_integral(cos.(grid), h)
    @test F[1] == 0
    @test maximum(abs, F .- sin.(grid)) < 1.0e-8

    # Exact for quadratics: Simpson integrates the quadratic interpolant
    # exactly, including the three-point rule used on the first interval.
    grid = range(0, 1; length = 21)
    h = step(grid)
    F = cumulative_integral(grid .^ 2, h)
    @test maximum(abs, F .- (collect(grid) .^ 3 ./ 3)) < 1.0e-14

    # Fourth-order convergence beyond that: halving `h` cuts the error
    # by roughly 2⁴ = 16.
    function cubic_error(np)
        g = range(0, 1; length = np)
        return maximum(abs, cumulative_integral(collect(g) .^ 3, step(g)) .- (collect(g) .^ 4 ./ 4))
    end
    e1, e2 = cubic_error(21), cubic_error(41)
    @test e2 < e1 / 8
    @test e1 < 1.0e-5

    # Degenerate lengths.
    @test cumulative_integral([1.0], 0.1) == [0.0]
    @test cumulative_integral([1.0, 1.0], 0.5) ≈ [0.0, 0.5]
end

@testset "time_ordered_integral" begin
    # For f ≡ 1 the bracket reduces to the Taylor coefficient
    # [f…f] = (-i)ⁿ (t - t₀)ⁿ / n!.
    Δ = 1.3
    for n in 1:5
        got = time_ordered_integral(fill(t -> 1.0, n), 0.0, Δ)
        @test got ≈ (-im)^n * Δ^n / factorial(n) rtol = 1.0e-7
    end

    # The empty bracket is 1.
    @test time_ordered_integral((), 0.0, 1.0) ≈ 1.0

    # A vanishing interval gives zero at every order.
    @test time_ordered_integral([t -> 1.0], 0.7, 0.7) ≈ 0.0 atol = 1.0e-14

    # Factoring property [f g] + [g f] = [f][g] (Vanthilt et al., Sec. V).
    f = t -> sin(2t) + 0.3
    g = t -> exp(-0.4t)
    t0, t1 = 0.0, 1.1
    fg = time_ordered_integral([f, g], t0, t1)
    gf = time_ordered_integral([g, f], t0, t1)
    @test fg + gf ≈ time_ordered_integral([f], t0, t1) * time_ordered_integral([g], t0, t1) rtol =
        1.0e-7

    # First-order bracket against a closed form: f(t) = t on [0, 2]
    # gives [f] = -i ∫₀² t dt = -2i.
    @test time_ordered_integral([identity], 0.0, 2.0) ≈ -2im rtol = 1.0e-9

    # Second-order bracket against a closed form:
    # (-i)² ∫₀ᵀ dt₁ ∫₀^{t₁} dt₂ t₁ t₂ = -T⁴/8.
    T = 1.7
    @test time_ordered_integral([identity, identity], 0.0, T) ≈ -T^4 / 8 rtol = 1.0e-7

    # The time ordering is not symmetric under swapping the functions.
    @test !isapprox(fg, gf; rtol = 1.0e-3)

    # Refining the grid converges.
    coarse = time_ordered_integral([f, g], t0, t1; npoints = 33)
    fine = time_ordered_integral([f, g], t0, t1; npoints = 2049)
    @test abs(coarse - fine) < 1.0e-6
end
