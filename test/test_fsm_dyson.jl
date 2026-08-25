using ITensorMPS
using ITensors
using LinearAlgebra
using TDVPlus
using TDVPlus: block_mpo, to_mpo, is_first_degree, virtualdim, rewire,
    concat_product, dyson_mpo_fsm, dyson_block_mpo, compress_columns,
    equivalent_column_groups, LevelTag, LEVEL_ONE,
    n_twos, n_threes, three_subscripts, strip_ones, is_start, dyson_mpo
using Test

isdefined(@__MODULE__, :dense_matrix) || include("dense_reference.jl")

# Densifying an n-site MPO forces a fresh compilation for each distinct
# `n`, because NDTensors specializes on tensor order: measured at ~8s for
# n = 4 rising to ~16s for n = 10, against ~0.0003s once warm. The cost
# is therefore per *chain length used anywhere in the suite*, not per
# call. So everything here reuses n = 6, the length the Dyson/Magnus and
# CFET tests already densify at, and the one test that genuinely needs a
# second length is the only place that pays for one.
const FSM_N = 6

function _fsm_model(n = FSM_N)
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

@testset "FSM level labels" begin
    l = [LEVEL_ONE, LevelTag(2, 1), LevelTag(3, 2), LevelTag(3, 1)]
    @test n_twos(l) == 1
    @test n_threes(l) == 2
    @test three_subscripts(l) == [2, 1]
    @test strip_ones(l) == [LevelTag(2, 1), LevelTag(3, 2), LevelTag(3, 1)]
    @test !is_start(l)
    @test is_start([LEVEL_ONE, LEVEL_ONE])
    # The all-1s level is what Algorithm 2 reroutes into, so it must
    # never itself satisfy the removal condition.
    @test n_threes([LEVEL_ONE, LEVEL_ONE]) == 0
end

@testset "BlockMPO round-trip" begin
    n = FSM_N
    sites = siteinds("S=1/2", n)
    zz = OpSum()
    for j in 1:(n - 1)
        zz += 1.0, "Sz", j, "Sz", j + 1
    end
    x = OpSum()
    for j in 1:n
        x += 0.7, "Sx", j
    end
    # An XXZ chain gives a chi > 1 middle block, which is what exercises
    # the reversed-index mapping; the onsite term gives the chi == 0 edge
    # case. Those two are the whole space of shapes worth checking.
    xxz = OpSum()
    for j in 1:(n - 1)
        xxz += 0.5, "S+", j, "S-", j + 1
        xxz += 0.5, "S-", j, "S+", j + 1
        xxz += 1.0, "Sz", j, "Sz", j + 1
    end

    for os in (zz, x, xxz)
        H = MPO(os, sites)
        B = block_mpo(H)
        @test is_first_degree(B)
        A1 = dense_matrix(H, sites)
        A2 = dense_matrix(to_mpo(B), sites)
        @test norm(A1 - A2) / norm(A1) < 1.0e-12
    end
end

@testset "rewire gives one 3-level per channel" begin
    sites, Hzz, Hx = _fsm_model()
    ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])
    H = rewire(ch)

    @test is_first_degree(H)
    threes = [l for l in H.levels if n_threes(l) == 1]
    @test length(threes) == 2
    @test sort([three_subscripts(l)[1] for l in threes]) == [1, 2]
    @test count(is_start, H.levels) == 1
    @test is_start(H.levels[1])
end

@testset "concat_product multiplies the virtual spaces" begin
    _, Hzz, Hx = _fsm_model()
    A = block_mpo(Hzz; channel = 1)
    B = block_mpo(Hx; channel = 2)
    P = concat_product(A, B)
    @test virtualdim(P) == virtualdim(A) * virtualdim(B)
    @test length(first(P.levels)) == 2
end

@testset "Dyson FSM converges at the advertised order" begin
    sites, Hzz, Hx = _fsm_model()
    Hd = dense_matrix(+(Hzz, Hx; alg = "directsum"), sites)

    # An order-N Dyson MPO is accurate through dtᴺ, so its error falls as
    # dt^(N+1). Checked against an independent dense exponential, never
    # against the same construction at a finer step. Orders 1 and 2 pin
    # the behaviour down; order 3 costs noticeably more for the same
    # statement.
    for order in 1:2
        errs = Float64[]
        for dt in (0.1, 0.05)
            ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])
            U = dense_matrix(dyson_mpo_fsm(ch, 0.0, dt; order, cutoff = 0.0), sites)
            Uex = exp(-im * dt * Hd)
            push!(errs, norm(U - Uex) / norm(Uex))
        end
        @test log2(errs[1] / errs[2]) > order + 0.7
    end
end

@testset "the default cutoff does not break convergence" begin
    # Regression: `dyson_mpo_fsm` originally defaulted to cutoff = 1e-12,
    # which looks tight but is not, because the construction is exact
    # and the truncation error lands directly on the result. Measured on
    # an XXZ chain (chi = 3), that default made order-3 convergence
    # invert (rate -0.31) once dt was small enough for the true step
    # error to drop below the truncation floor. cutoff = 1e-14 (now the
    # default) does not touch the convergence rate.
    n = FSM_N
    sites = siteinds("S=1/2", n)
    xxz = OpSum()
    for j in 1:(n - 1)
        xxz += 0.5, "S+", j, "S-", j + 1
        xxz += 0.5, "S-", j, "S+", j + 1
        xxz += 1.0, "Sz", j, "Sz", j + 1
    end
    H = MPO(xxz, sites)
    Hd = dense_matrix(H, sites)

    errs = Float64[]
    for dt in (0.1, 0.05, 0.025)
        ch = DrivingChannels([(1.0, H)])
        U = dense_matrix(dyson_mpo_fsm(ch, 0.0, dt; order = 3), sites)
        push!(errs, norm(U - exp(-im * dt * Hd)) / norm(exp(-im * dt * Hd)))
    end
    rates = [log2(errs[i] / errs[i + 1]) for i in 1:2]
    @test all(r -> r > 3.7, rates)
end

@testset "equivalent-column compression is exact" begin
    sites, Hzz, Hx = _fsm_model()

    for order in 2:3
        ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])
        raw = dyson_block_mpo(ch, 0.0, 0.1; order, compress = false)
        small = compress_columns(raw)

        # It must shrink ...
        @test virtualdim(small) < virtualdim(raw)
        # ... and encode exactly the same operator. This is the exact
        # step of Sec. VI A, so machine precision is the right bar.
        A1 = dense_matrix(to_mpo(raw), sites)
        A2 = dense_matrix(to_mpo(small), sites)
        @test norm(A1 - A2) / norm(A1) < 1.0e-12
    end

    # Levels are grouped by history, i.e. by their label with the 1s
    # removed -- a 1 marks a factor that has not started, so sliding it
    # through cannot change what came before.
    ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])
    raw = dyson_block_mpo(ch, 0.0, 0.1; order = 3, compress = false)
    for g in equivalent_column_groups(raw)
        @test allequal(strip_ones(raw.levels[i]) for i in g)
    end
end

@testset "Dyson FSM is size-extensive" begin
    # The defining property of the paper's construction: the error per
    # site stays put as the chain grows, where the direct construction of
    # `dyson_mpo` degrades sharply. Two lengths are enough to show the
    # divergence -- the effect is orders of magnitude, not marginal.
    dt = 0.1
    order = 2
    fsm, direct = Float64[], Float64[]

    # n = 4 is the only extra chain length this file introduces; n = 6 is
    # already warm from the tests above.
    for n in (4, FSM_N)
        sites, Hzz, Hx = _fsm_model(n)
        Hd = dense_matrix(+(Hzz, Hx; alg = "directsum"), sites)

        psi0 = MPS(sites, j -> "Up")
        v0 = dense_vector(psi0, sites)
        ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])

        vex = exp(-im * dt * Hd) * v0
        uf = apply(dyson_mpo_fsm(ch, 0.0, dt; order), psi0; cutoff = 1.0e-14)
        ud = apply(dyson_mpo(ch, 0.0, dt; order), psi0; cutoff = 1.0e-14)

        push!(fsm, infidelity(vex, dense_vector(uf, sites)) / n)
        push!(direct, infidelity(vex, dense_vector(ud, sites)) / n)
    end

    # Per-site error roughly flat for the FSM construction ...
    @test fsm[2] / fsm[1] < 2.0
    # ... and growing for the direct one, which is the bug Tier 3 fixes.
    @test direct[2] / direct[1] > 3.0
    @test direct[2] > 50 * fsm[2]
end

@testset "dyson_evolve inherits size-extensivity from the swap" begin
    # `dyson_evolve` used to build each step with the direct construction
    # `dyson_mpo`; it now uses `dyson_mpo_fsm`. This checks the swap
    # through the actual user-facing driver -- multiple steps,
    # normalization, apply truncation -- rather than only the single-step
    # MPO tested above.
    T, dt = 0.3, 0.1
    order = 2
    per_site = Float64[]
    for n in (4, FSM_N)
        sites, Hzz, Hx = _fsm_model(n)
        Hd = dense_matrix(+(Hzz, Hx; alg = "directsum"), sites)
        psi0 = MPS(sites, j -> "Up")
        v0 = dense_vector(psi0, sites)
        ch = DrivingChannels([(1.0, Hzz), (1.0, Hx)])

        evolved = dyson_evolve(ch, psi0, 0.0, T; dt, order)
        vex = exp(-im * T * Hd) * v0
        push!(per_site, infidelity(vex, dense_vector(evolved, sites)) / n)
    end
    @test per_site[2] / per_site[1] < 2.0
end

@testset "compression handles a chi > 1 block" begin
    # Regression: `Level` originally labelled every state of a channel's
    # in-progress block as plain `2_a`, so the column compression merged
    # states that are not interchangeable. Every chi <= 1 model misses
    # this; XXZ has chi = 3 and catches it.
    n = FSM_N
    sites = siteinds("S=1/2", n)
    xxz = OpSum()
    for j in 1:(n - 1)
        xxz += 0.5, "S+", j, "S-", j + 1
        xxz += 0.5, "S-", j, "S+", j + 1
        xxz += 1.0, "Sz", j, "Sz", j + 1
    end
    Hxxz = MPO(xxz, sites)

    # Merging states that are not interchangeable changes the operator,
    # so comparing the compressed build against the uncompressed one is
    # the sharp test -- and needs no tolerance on the physics.
    for order in 1:2
        ch = DrivingChannels([(1.0, Hxxz)])
        raw = dyson_block_mpo(ch, 0.0, 0.02; order, compress = false)
        small = compress_columns(raw)
        @test virtualdim(small) <= virtualdim(raw)

        A1 = dense_matrix(to_mpo(raw), sites)
        A2 = dense_matrix(to_mpo(small), sites)
        @test norm(A1 - A2) / norm(A1) < 1.0e-12
    end
end

@testset "Dyson FSM conserves quantum numbers" begin
    n = FSM_N
    sites = siteinds("S=1/2", n; conserve_qns = true)
    zz = OpSum()
    for j in 1:(n - 1)
        zz += 1.0, "Sz", j, "Sz", j + 1
    end
    hop = OpSum()
    for j in 1:(n - 1)
        hop += 0.5, "S+", j, "S-", j + 1
        hop += 0.5, "S-", j, "S+", j + 1
    end
    Hzz, Hhop = MPO(zz, sites), MPO(hop, sites)

    # The block structure survives extraction and reconstruction. Checked
    # through the action on a state, since a QN MPO cannot be densified
    # as cheaply as a plain one.
    psi = random_mps(sites, j -> isodd(j) ? "Up" : "Dn"; linkdims = 4)
    for H in (Hzz, Hhop)
        B = block_mpo(H)
        @test !isnothing(B.qns)
        a = apply(H, psi; cutoff = 1.0e-14)
        b = apply(to_mpo(B), psi; cutoff = 1.0e-14)
        @test norm(a - b) / norm(a) < 1.0e-12
    end

    psi0 = MPS(sites, j -> isodd(j) ? "Up" : "Dn")
    ch = DrivingChannels([(1.0, Hzz), (t -> cos(2.0t), Hhop)])
    for order in 1:2
        evolved = apply(dyson_mpo_fsm(ch, 0.0, 0.05; order), psi0; cutoff = 1.0e-12)
        @test flux(evolved) == flux(psi0)
        @test isapprox(norm(evolved), 1.0; atol = 1.0e-2)
    end
end

@testset "Dyson FSM matches a time-dependent reference" begin
    sites, Hzz, Hx = _fsm_model()
    f1 = t -> 1.0
    f2 = t -> cos(3.0 * t)
    Hmats = [dense_matrix(Hzz, sites), dense_matrix(Hx, sites)]

    psi0 = MPS(sites, j -> "Up")
    v0 = dense_vector(psi0, sites)
    dt = 0.1
    vref = dense_rk4_reference(Hmats, [f1, f2], v0, 0.0, dt; nsub = 400)

    errs = Float64[]
    for order in 1:2
        ch = DrivingChannels([(f1, Hzz), (f2, Hx)])
        U = dense_matrix(dyson_mpo_fsm(ch, 0.0, dt; order, cutoff = 0.0), sites)
        push!(errs, infidelity(vref, U * v0))
    end
    # Raising the order strictly improves the step for a genuinely
    # time-dependent drive.
    @test errs[2] < errs[1]
end
