"""
    DrivingChannels(operators::Vector{MPO}, drivings::Vector, sites)
    DrivingChannels(H1 => f1, H2 => f2, ...)
    DrivingChannels([H1 => f1, H2 => f2, ...])

A time-dependent Hamiltonian in the "driving channel" form of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597):

```math
H(t) = \\sum_a f_a(t) \\, H^{(a)}
```

where each `H^{(a)}` is a time-*independent* MPO and each `f_a` is a
scalar driving function. This is the decomposition that both the Dyson
and Magnus MPO constructions are built on: it separates the operator
content (the `H^{(a)}`, which can be multiplied and commuted once and for
all) from the time dependence (the `f_a`, which enter only through the
scalar time-ordered integrals of [`time_ordered_integral`](@ref)).

Calling `channels(t)` assembles `H(t)` as an ordinary MPO, which is
mainly useful as a reference for testing.

# Examples

```julia
sites = siteinds("S=1/2", 8)
Hzz = MPO(OpSum() + ..., sites)   # static coupling
Hx  = MPO(OpSum() + ..., sites)   # transverse field
ramp = Ramp(SmoothstepRamp(), 0.0, 1.0, 0.0, 2.0)
channels = DrivingChannels(Hzz => one, Hx => ramp)
```
"""
struct DrivingChannels{F <: AbstractVector, S}
    operators::Vector{MPO}
    drivings::F
    sites::S
end

# The site indices an MPO was *built from*. `firstsiteinds` returns the
# MPO's own unprimed leg, which for a QN-conserving MPO carries the
# opposite arrow to the index the user passed to `siteinds`; building an
# identity MPO from those flipped indices yields something that cannot be
# contracted with the Hamiltonians. `dag` restores the original arrows,
# and is a no-op without QNs.
_ket_siteinds(H::MPO) = dag(firstsiteinds(H))

function DrivingChannels(operators::AbstractVector{MPO}, drivings::AbstractVector)
    if length(operators) != length(drivings)
        throw(
            ArgumentError(
                "got $(length(operators)) operators but $(length(drivings)) driving functions; they must match."
            )
        )
    end
    isempty(operators) && throw(ArgumentError("`DrivingChannels` needs at least one channel."))
    sites = _ket_siteinds(first(operators))
    for (a, H) in enumerate(operators)
        if _ket_siteinds(H) != sites
            throw(ArgumentError("channel $a has site indices differing from channel 1."))
        end
    end
    return DrivingChannels(collect(MPO, operators), drivings, sites)
end

"""
Anything that names one channel: a `(f, H)` or `(H, f)` tuple, or an
`H => f` / `f => H` pair.
"""
const ChannelSpec = Union{Pair, Tuple{Any, Any}}

# The MPO and the driving function are told apart by type, so callers do
# not have to remember an argument order.
function _channel_pair(spec::ChannelSpec)
    a, b = spec isa Pair ? (first(spec), last(spec)) : spec
    a isa MPO && !(b isa MPO) && return (a, b)
    b isa MPO && !(a isa MPO) && return (b, a)
    return throw(
        ArgumentError(
            "each channel must pair exactly one MPO with one driving function; got ($(typeof(a)), $(typeof(b)))."
        )
    )
end
function _channel_pair(spec)
    return throw(
        ArgumentError(
            "expected each channel to be a `(f, H)` tuple or an `H => f` pair, got a $(typeof(spec))."
        )
    )
end

"""
    DrivingChannels([(f1, H1), (f2, H2), ...])
    DrivingChannels(H1 => f1, H2 => f2, ...)

Build the channel decomposition from a list of `(driving function, MPO)`
tuples or pairs, in either order.
"""
function DrivingChannels(specs::AbstractVector)
    isempty(specs) && throw(ArgumentError("`DrivingChannels` needs at least one channel."))
    prs = map(_channel_pair, specs)
    return DrivingChannels(MPO[p[1] for p in prs], [p[2] for p in prs])
end
DrivingChannels(specs::ChannelSpec...) = DrivingChannels(collect(specs))

"""
    nchannels(channels::DrivingChannels)

The number of driving channels.
"""
nchannels(channels::DrivingChannels) = length(channels.operators)

ITensorMPS.siteinds(channels::DrivingChannels) = channels.sites

"""
    identity_mpo(channels::DrivingChannels)

The identity MPO on the sites of `channels`, i.e. the zeroth-order term
of the Dyson series.
"""
identity_mpo(channels::DrivingChannels) = MPO(channels.sites, "Id")

"""
    (channels::DrivingChannels)(t; cutoff, maxdim)

Assemble `H(t) = Σₐ fₐ(t) H^{(a)}` as an MPO.
"""
function (channels::DrivingChannels)(t; cutoff = 1.0e-14, maxdim = typemax(Int))
    terms = [channels.drivings[a](t) * channels.operators[a] for a in 1:nchannels(channels)]
    length(terms) == 1 && return only(terms)
    return sum(terms; cutoff, maxdim)
end

"""
    commutator(A::MPO, B::MPO; cutoff, maxdim)

The commutator `A*B - B*A` as an MPO.

For two extensive Hamiltonians the commutator is automatically free of
the "disjoint" contributions that spoil size-extensivity, since those
terms are identical in `A*B` and `B*A` and cancel; see Sec. III of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597).
"""
function commutator(A::MPO, B::MPO; cutoff = 1.0e-14, maxdim = typemax(Int))
    AB = apply(A, B; cutoff, maxdim)
    BA = apply(B, A; cutoff, maxdim)
    return +(AB, -BA; cutoff, maxdim)
end
