using ITensorMPS
using ITensors
using ITensorMPSExtended
using LinearAlgebra
using Test

function te_model(n = 6)
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

# Fidelity between two MPS. These drivers do not renormalize, so the norm
# drifts by ~1e-11 under truncation; comparing raw `abs(inner(a, b))`
# against 1 would fail for states that are in fact identical.
fid(a, b) = abs(inner(a, b)) / (norm(a) * norm(b))

@testset "time_evolve dispatches to each driver" begin
    sites, Hzz, Hx = te_model()
    g = t -> sin(4t) + 0.5
    ch = DrivingChannels(Hzz => one, Hx => g)
    ψ0 = MPS(ComplexF64, sites, "Up")
    T, ns = 0.3, 4
    st = (; cutoff = 1.0e-13, maxdim = 128)
    op = (; cutoff = 1.0e-13, maxdim = 128)

    # Magnus is the default algorithm.
    ψ_default = time_evolve(ch, ψ0, 0.0, T; nsteps = ns, cutoff = st.cutoff, maxdim = st.maxdim,
        operator_cutoff = op.cutoff, operator_maxdim = op.maxdim)
    ψ_magnus = magnus_evolve(
        ch, ψ0, 0.0, T; nsteps = ns, order = 2,
        generator_kwargs = op, tdvp_kwargs = st
    )
    @test fid(ψ_default, ψ_magnus) ≈ 1.0 atol = 1.0e-10

    # Explicit magnus, dyson, piecewise_constant all match their drivers.
    ψ_dyson = time_evolve(ch, ψ0, 0.0, T; alg = "dyson", nsteps = ns,
        cutoff = st.cutoff, maxdim = st.maxdim,
        operator_cutoff = op.cutoff, operator_maxdim = op.maxdim)
    ψ_dyson_direct = dyson_evolve(
        ch, ψ0, 0.0, T; nsteps = ns, order = 2, mpo_kwargs = op, apply_kwargs = st
    )
    @test fid(ψ_dyson, ψ_dyson_direct) ≈ 1.0 atol = 1.0e-10

    ψ_pc = time_evolve(ch, ψ0, 0.0, T; alg = "piecewise_constant", nsteps = ns,
        cutoff = st.cutoff, maxdim = st.maxdim,
        operator_cutoff = op.cutoff, operator_maxdim = op.maxdim)
    ψ_pc_direct = piecewise_constant_tdvp(
        ch, ψ0, 0.0, T; nsteps = ns,
        operator_cutoff = op.cutoff, operator_maxdim = op.maxdim, tdvp_kwargs = st
    )
    @test fid(ψ_pc, ψ_pc_direct) ≈ 1.0 atol = 1.0e-10

    # All three describe the same physics, so they agree to their accuracy.
    @test fid(ψ_magnus, ψ_pc) > 0.999
    @test fid(ψ_magnus, ψ_dyson) > 0.999
end

@testset "piecewise_constant_tdvp from DrivingChannels matches H0/Ht form" begin
    sites, Hzz, Hx = te_model()
    g = t -> cos(t) + 0.3
    ch = DrivingChannels(Hzz => one, Hx => g)
    ψ0 = MPS(ComplexF64, sites, "Up")
    st = (; cutoff = 1.0e-13, maxdim = 128)

    ψ_ch = piecewise_constant_tdvp(ch, ψ0, 0.0, 0.3; nsteps = 4, tdvp_kwargs = st)
    ψ_pair = piecewise_constant_tdvp(
        Hzz, t -> g(t) * Hx, ψ0, 0.0, 0.3; nsteps = 4, tdvp_kwargs = st
    )
    @test fid(ψ_ch, ψ_pair) ≈ 1.0 atol = 1.0e-10
end

@testset "channels given as a plain list of tuples" begin
    sites, Hzz, Hx = te_model()
    g = t -> sin(4t) + 0.5
    ψ0 = MPS(ComplexF64, sites, "Up")
    T, ns = 0.3, 4

    explicit = DrivingChannels(Hzz => one, Hx => g)
    ψ_explicit = time_evolve(explicit, ψ0, 0.0, T; nsteps = ns)

    # (f, H) tuples, the requested form.
    ψ_tuples = time_evolve([(one, Hzz), (g, Hx)], ψ0, 0.0, T; nsteps = ns)
    @test fid(ψ_explicit, ψ_tuples) ≈ 1.0 atol = 1.0e-10

    # (H, f) tuples: order within a channel should not matter.
    ψ_flipped = time_evolve([(Hzz, one), (Hx, g)], ψ0, 0.0, T; nsteps = ns)
    @test fid(ψ_explicit, ψ_flipped) ≈ 1.0 atol = 1.0e-10

    # Pairs in either direction too.
    ψ_pairs = time_evolve([Hzz => one, g => Hx], ψ0, 0.0, T; nsteps = ns)
    @test fid(ψ_explicit, ψ_pairs) ≈ 1.0 atol = 1.0e-10

    # Works for every algorithm, and with an explicit time grid.
    for alg in ("magnus", "dyson", "piecewise_constant")
        @test time_evolve([(one, Hzz), (g, Hx)], ψ0, range(0.0, T; length = 3); alg) isa MPS
    end

    # DrivingChannels accepts the same forms directly.
    @test nchannels(DrivingChannels([(one, Hzz), (g, Hx)])) == 2
    @test nchannels(DrivingChannels((one, Hzz), (g, Hx))) == 2

    # Malformed channels are rejected with a useful message.
    @test_throws ArgumentError DrivingChannels([(one, g)])          # no MPO
    @test_throws ArgumentError DrivingChannels([(Hzz, Hx)])         # two MPOs
    @test_throws ArgumentError DrivingChannels([(one, Hzz, g)])     # not a pair
    @test_throws ArgumentError DrivingChannels(Any[])
end

@testset "time_evolve argument handling" begin
    sites, Hzz, Hx = te_model(4)
    ch = DrivingChannels(Hzz => one, Hx => (t -> 1.0))
    ψ0 = MPS(ComplexF64, sites, "Up")

    # String, Symbol and Algorithm all select the same method.
    a = time_evolve(ch, ψ0, 0.0, 0.1; alg = "magnus", nsteps = 2)
    b = time_evolve(ch, ψ0, 0.0, 0.1; alg = :magnus, nsteps = 2)
    @test fid(a, b) ≈ 1.0 atol = 1.0e-10

    # Unknown algorithms are rejected by name, with a message that names
    # the offending algorithm and the valid ones. (Building that message
    # must not itself throw — `String(::Algorithm)` is unavailable.)
    err = try
        time_evolve(ch, ψ0, 0.0, 0.1; alg = "trotter", nsteps = 2)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("trotter", sprint(showerror, err))
    @test occursin("magnus", sprint(showerror, err))
    @test_throws ArgumentError time_evolve(ch, ψ0, 0.0, 0.1; alg = :trotter, nsteps = 2)

    # `order` is meaningless when the step is not expanded.
    @test_throws ArgumentError time_evolve(
        ch, ψ0, 0.0, 0.1; alg = "piecewise_constant", nsteps = 2, order = 2
    )
    # ...but is accepted by the expansion methods.
    @test time_evolve(ch, ψ0, 0.0, 0.1; alg = "magnus", nsteps = 2, order = 1) isa MPS
    @test time_evolve(ch, ψ0, 0.0, 0.1; alg = "dyson", nsteps = 2, order = 1) isa MPS

    # An explicit time grid works as well as t_start/t_stop.
    grid = range(0.0, 0.1; length = 3)
    @test time_evolve(ch, ψ0, grid; alg = "magnus") isa MPS

    # alg_kwargs reach the underlying driver.
    steps = Int[]
    time_evolve(
        ch, ψ0, 0.0, 0.1; alg = "dyson", nsteps = 2,
        step_observer! = (; step, t_start, t_stop, state) -> push!(steps, step)
    )
    @test steps == [1, 2]
    @test time_evolve(
        ch, ψ0, 0.0, 0.1; alg = "dyson", nsteps = 2, alg_kwargs = (; normalize = false)
    ) isa MPS
end
