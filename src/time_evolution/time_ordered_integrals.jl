"""
    cumulative_integral(y::AbstractVector, h)

Cumulative integral of samples `y` on a uniform grid of spacing `h`,
returning a vector `F` with `F[i] ≈ ∫_{x_1}^{x_i} y(x) dx` and `F[1] == 0`.

Uses the composite Simpson rule (error `O(h⁴)`), with the standard
three-point formula for the first sub-interval so that odd-indexed
points are also fourth-order accurate.
"""
function cumulative_integral(y::AbstractVector, h)
    n = length(y)
    T = typeof(zero(eltype(y)) * zero(h))
    F = zeros(T, n)
    n <= 1 && return F
    if n == 2
        F[2] = h * (y[1] + y[2]) / 2
        return F
    end
    F[2] = h * (5y[1] + 8y[2] - y[3]) / 12
    for i in 3:n
        F[i] = F[i - 2] + h * (y[i - 2] + 4y[i - 1] + y[i]) / 3
    end
    return F
end

"""
    time_ordered_integral(fs, t0, t; npoints = 1025)

The time-ordered integral of the driving functions `fs = (f₁, …, fₙ)`
over `[t0, t]`, written `[f₁ f₂ … fₙ]` in
[Vanthilt et al.](https://arxiv.org/abs/2605.21597):

```math
[f_1 f_2 … f_n] = (-i)^n ∫_{t_0}^{t} dt_1 ∫_{t_0}^{t_1} dt_2 ⋯
                  ∫_{t_0}^{t_{n-1}} dt_n \\; f_1(t_1) f_2(t_2) ⋯ f_n(t_n)
```

Note the ordering convention: `t₁ > t₂ > ⋯ > tₙ`, so `f₁` is evaluated at
the *latest* time. These are exactly the scalar prefactors multiplying the
operator strings `H^{(a₁)} H^{(a₂)} ⋯ H^{(aₙ)}` in the Dyson series, and
the prefactors of the nested commutators in the Magnus expansion. The
factors of `-i` are folded into the bracket, so no further factors of `-i`
are needed downstream.

The empty bracket `[]` (`n == 0`) is `1`.

The nested integrals are evaluated by repeated cumulative quadrature on a
uniform grid of `npoints` points, which costs `O(n · npoints)` rather than
the `O(npoints^n)` of naive nested quadrature. `npoints` is rounded up to
an odd number.

# Examples

For a constant driving function `f ≡ 1`, the bracket reduces to the
Taylor coefficient `(-i)ⁿ (t - t₀)ⁿ / n!`:

```julia
time_ordered_integral((one, one), 0.0, 1.0)  # ≈ -0.5
```
"""
function time_ordered_integral(fs, t0, t; npoints::Integer = 1025)
    n = length(fs)
    n == 0 && return one(ComplexF64)
    npoints = max(3, isodd(npoints) ? npoints : npoints + 1)
    grid = range(float(t0), float(t); length = npoints)
    h = step(grid)
    # `g` holds the running inner integral; it starts as g_{n+1} ≡ 1.
    g = ones(ComplexF64, npoints)
    for k in n:-1:1
        f = fs[k]
        y = ComplexF64[f(x) for x in grid] .* g
        g = cumulative_integral(y, h)
    end
    return (-im)^n * g[end]
end
