"""
    concat_product(B1::BlockMPO, B2::BlockMPO)

The product `B1 * B2` as a [`BlockMPO`](@ref), formed by taking the
tensor product of the virtual spaces and multiplying the single-site
operators. Virtual index `(i, j)` of the result carries the concatenated
level label `[B1.levels[i]; B2.levels[j]]`.

This is the plain concatenated product of the paper's Eq. (36) — the one
that is *not* a first-degree MPO, because it retains the `(1,3)` and
`(3,1)` levels connecting disjoint terms. That is deliberate:
Algorithm 2 starts from exactly this object and reroutes those levels,
which is what restores size-extensivity.

The label ordering follows the operator ordering: slot 1 comes from the
left factor, which acts at the *later* time, matching the convention of
[`time_ordered_integral`](@ref) where `f₁` is evaluated latest.
"""
function concat_product(B1::BlockMPO, B2::BlockMPO)
    length(B1) == length(B2) ||
        throw(ArgumentError("cannot multiply BlockMPOs of lengths $(length(B1)) and $(length(B2))."))
    d2 = virtualdim(B2)
    idx(i, j) = (i - 1) * d2 + j

    levels = Vector{Level}(undef, virtualdim(B1) * d2)
    for i in 1:virtualdim(B1), j in 1:d2
        levels[idx(i, j)] = vcat(B1.levels[i], B2.levels[j])
    end

    W = [Dict{Tuple{Int, Int}, ITensor}() for _ in 1:length(B1)]
    for n in eachindex(W)
        for ((r1, c1), op1) in B1.W[n], ((r2, c2), op2) in B2.W[n]
            _addop!(W[n], idx(r1, r2), idx(c1, c2), _opprod(op1, op2))
        end
    end

    # Flux is additive under the operator product, so a composite level
    # sits in the sum of its factors' sectors.
    qns = if hasqns(B1) && hasqns(B2)
        [B1.qns[i] + B2.qns[j] for i in 1:virtualdim(B1) for j in 1:d2]
    else
        nothing
    end

    return BlockMPO(B1.sites, W, levels, qns, idx(B1.vL, B2.vL), idx(B1.vR, B2.vR))
end

"""
    rewire(channels::DrivingChannels)

The rewired time-dependent Hamiltonian `H̃` of the paper's Eq. (63), as a
[`BlockMPO`](@ref): the driving channels are summed into one first-degree
MPO, but with a *separate* finished level `(3ₐ)` for each channel `a`
rather than the single shared level `(3)`.

That separation is the whole point. In the plain sum of Eq. (45) all
channels funnel into one finished level, so a column of `H̃^N` no longer
records *which* driving functions produced the operator string it holds.
With one `3ₐ` per channel, the subscripts of a column's `3`-levels spell
out the permutation `σ`, and hence exactly which time-ordered integral
`[f_σ(1) ⋯ f_σ(n₃)]` that column must be weighted by.

The driving functions themselves are deliberately *not* substituted here.
Eq. (63) writes `fₐ(t)` on the `D⁽ᵃ⁾` and `R⁽ᵃ⁾` entries, but those
factors are precisely what Algorithm 2 replaces by the integrated
brackets when it reroutes the column; carrying them numerically as well
would double-count the time dependence.
"""
function rewire(channels::DrivingChannels)
    nc = nchannels(channels)
    blocks = [block_mpo(channels.operators[a]; channel = a) for a in 1:nc]
    nsites = length(first(blocks))

    # Levels: (1), then each channel's 2-block, then one 3ₐ per channel.
    chis = [virtualdim(B) - 2 for B in blocks]
    two_offset = Vector{Int}(undef, nc)
    off = 1
    for a in 1:nc
        two_offset[a] = off
        off += chis[a]
    end
    three_index = [off + a for a in 1:nc]

    levels = Level[[LEVEL_ONE]]
    # `slot` runs over the states inside channel `a`'s block, in the same
    # order `remap` places them, so the labels stay in step with the
    # virtual indices they name.
    for a in 1:nc, i in 1:chis[a]
        push!(levels, [LevelTag(2, a, i)])
    end
    for a in 1:nc
        push!(levels, [LevelTag(3, a)])
    end

    W = [Dict{Tuple{Int, Int}, ITensor}() for _ in 1:nsites]
    for a in 1:nc
        B = blocks[a]
        d = virtualdim(B)
        # Map channel `a`'s local index to the combined index.
        function remap(i)
            i == 1 && return 1
            i == d && return three_index[a]
            return two_offset[a] + (i - 1)
        end
        for n in 1:nsites
            for ((r, c), op) in B.W[n]
                # The identity on the untouched level (1) is shared by
                # every channel; taking it from each would multiply it by
                # the number of channels.
                r == 1 && c == 1 && a != 1 && continue
                _addop!(W[n], remap(r), remap(c), op)
            end
        end
    end

    qns = if all(hasqns, blocks)
        q = [blocks[1].qns[1]]                       # the shared level (1)
        for a in 1:nc, i in 2:(virtualdim(blocks[a]) - 1)
            push!(q, blocks[a].qns[i])               # each channel's 2-block
        end
        for a in 1:nc
            push!(q, blocks[a].qns[end])             # each channel's 3ₐ
        end
        q
    else
        nothing
    end

    return BlockMPO(first(blocks).sites, W, levels, qns)
end

"""
    dyson_block_mpo(channels, t0, t; order, npoints, prefactor)

The `order`-th order Dyson MPO of the paper's Algorithm 2, as a
size-extensive [`BlockMPO`](@ref).

Starting from `H̃^N` (`N = order`), every level whose label contains no
in-progress `2` but at least one finished `3ₐ` represents a completed
operator string. Its column is weighted by the time-ordered integral
named by its `3`-subscripts, together with the combinatorial factor
`n₃!(N-n₃)!/N!` that corrects for the `N!/(n₃!(N-n₃)!)` equivalent
levels holding the same string, and added back into the level `(1)`
column. Those levels are then removed.

Removing the finished levels destroys the upper-triangular structure,
and that is what makes the result size-extensive: the operator both
enters and exits at level `(1)`, so the MPO can be applied to an MPS
directly. Its accuracy no longer degrades with chain length, unlike the
direct construction of [`dyson_terms`](@ref).

The construction is *exact* — no truncation is performed here, since the
entries are single-site operators. Truncation enters only when the result
is applied to a state.
"""
function dyson_block_mpo(
        channels::DrivingChannels, t0, t;
        order::Integer,
        npoints::Integer = 1025,
        prefactor = -im,
        compress::Bool = true
    )
    order >= 1 || throw(ArgumentError("`order` must be at least 1, got $order."))

    H = rewire(channels)
    O = H
    for _ in 2:order
        O = concat_product(O, H)
    end

    N = order
    removable = [i for i in 1:virtualdim(O)
        if n_twos(O.levels[i]) == 0 && n_threes(O.levels[i]) >= 1]

    # Every merge reads a column and writes column 1, and column 1 is
    # never itself merged, so the merges do not interact and can be
    # applied in any order before the levels are dropped.
    for l in removable
        sigma = three_subscripts(O.levels[l])
        n3 = length(sigma)
        bracket = time_ordered_integral(
            [channels.drivings[a] for a in sigma], t0, t; npoints, prefactor
        )
        coeff = bracket * factorial(big(n3)) * factorial(big(N - n3)) / factorial(big(N))
        iszero(coeff) && continue
        c = ComplexF64(coeff)
        for n in eachindex(O.W)
            for r in 1:virtualdim(O)
                op = _getop(O.W[n], r, l)
                isnothing(op) && continue
                _addop!(O.W[n], r, 1, c * op)
            end
        end
    end

    # Rerouting moved the exit onto level (1): the operator now both
    # enters and leaves there, which is what makes it extensive.
    O = BlockMPO(O.sites, O.W, O.levels, O.qns, O.vL, O.vL)
    O = _drop_levels(O, removable)
    # Exact, so it is on by default: it changes the bond dimension and
    # nothing else.
    return compress ? compress_columns(O) : O
end

# Delete a set of virtual levels and renumber the survivors.
function _drop_levels(B::BlockMPO, drop)
    dropset = Set(drop)
    keep = [i for i in 1:virtualdim(B) if !(i in dropset)]
    renumber = Dict(old => new for (new, old) in enumerate(keep))

    W = [Dict{Tuple{Int, Int}, ITensor}() for _ in eachindex(B.W)]
    for n in eachindex(B.W)
        for ((r, c), op) in B.W[n]
            (haskey(renumber, r) && haskey(renumber, c)) || continue
            W[n][(renumber[r], renumber[c])] = op
        end
    end

    (haskey(renumber, B.vL) && haskey(renumber, B.vR)) ||
        throw(ArgumentError("a boundary level was dropped; this is a bug in the caller."))
    qns = hasqns(B) ? B.qns[keep] : nothing
    return BlockMPO(B.sites, W, B.levels[keep], qns, renumber[B.vL], renumber[B.vR])
end

"""
    dyson_mpo_fsm(channels, t0, t; order, cutoff, maxdim, npoints, prefactor)

The size-extensive Dyson MPO of [`dyson_block_mpo`](@ref), contracted
into an ordinary `ITensorMPS` MPO ready to apply to a state.

`cutoff` and `maxdim` truncate the resulting MPO after construction.
"""
function dyson_mpo_fsm(
        channels::DrivingChannels, t0, t;
        order::Integer = 2,
        cutoff = 1.0e-12,
        maxdim = typemax(Int),
        npoints::Integer = 1025,
        prefactor = -im,
        compress::Bool = true
    )
    B = dyson_block_mpo(channels, t0, t; order, npoints, prefactor, compress)
    U = to_mpo(B)
    return truncate(U; cutoff, maxdim)
end
