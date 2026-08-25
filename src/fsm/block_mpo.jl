"""
    BlockMPO(sites, W, levels)

An MPO held as an explicit operator-valued matrix per site, together with
the finite-state-machine [`Level`](@ref) labelling each virtual index.

This is the representation the constructions of
[Vanthilt et al.](https://arxiv.org/abs/2605.21597) operate on, and which
`ITensorMPS` does not expose: `MPO(::OpSum, sites)` compresses the Jordan
block structure away behind opaque link indices. Keeping the levels
explicit is what makes the Dyson construction of Algorithm 2 — which must
ask how many `2`s and `3ₐ`s a virtual index carries — expressible at all.

Virtual indices are stored in the paper's ordering, so index `1` is the
level `(1)` where no term has started and index `end` is the level `(3)`
where a term has finished. The operator is recovered by contracting with
the boundary vectors `v_L = (1, 0, …, 0)` and `v_R = (0, …, 0, 1)`, i.e.
by taking row `1` of the first site and column `end` of the last.

`W[n]` maps `(row, column)` to the single-site operator on that
transition; absent keys are zero. A *first-degree* MPO in this form is
upper triangular, and is not size-extensive — see
[`is_first_degree`](@ref).

`vL` and `vR` record which virtual indices the boundary vectors select.
They are not always the two ends: the Dyson construction reroutes every
finished level back into level `(1)`, so its output both enters *and*
exits at index `1`, and reading it off at index `end` would silently give
the wrong operator.
"""
struct BlockMPO{S <: Index}
    sites::Vector{S}
    W::Vector{Dict{Tuple{Int, Int}, ITensor}}
    levels::Vector{Level}
    vL::Int
    vR::Int
end

# A first-degree MPO runs from the level where nothing has started to the
# level where everything has finished.
function BlockMPO(sites, W, levels)
    return BlockMPO(sites, W, levels, 1, length(levels))
end

"""
    virtualdim(B::BlockMPO)

The dimension of `B`'s virtual index, i.e. the number of FSM levels.
"""
virtualdim(B::BlockMPO) = length(B.levels)

Base.length(B::BlockMPO) = length(B.sites)

"""
    is_first_degree(B::BlockMPO)

Whether `B` is upper triangular in its virtual levels, the defining
property of a first-degree MPO. Such an MPO represents an extensive
*Hamiltonian* as a bookkeeping device but is not itself an extensive
operator: applying it to an MPS gives a state that cannot be normalized.
The Dyson and Taylor constructions exist precisely to turn it back into
a size-extensive one.
"""
function is_first_degree(B::BlockMPO)
    return all(all(r <= c for (r, c) in keys(W)) for W in B.W)
end

# The product of two single-site operators, each carrying `s'` (out) and
# `s` (in). Priming `a` lifts its legs to `(s'', s')`, so contracting
# against `b`'s `s'` composes them; the result is mapped back to `(s', s)`.
function _opprod(a::ITensor, b::ITensor)
    return replaceprime(prime(a) * b, 2 => 1)
end

_getop(W::Dict{Tuple{Int, Int}, ITensor}, r::Int, c::Int) = get(W, (r, c), nothing)

function _addop!(W::Dict{Tuple{Int, Int}, ITensor}, r::Int, c::Int, op::ITensor)
    old = get(W, (r, c), nothing)
    W[(r, c)] = isnothing(old) ? op : old + op
    return W
end

"""
    block_mpo(H::MPO; channel = 1)

Read the first-degree block structure `{L, R, A, D}` off an `ITensorMPS`
MPO built from an `OpSum`, as a [`BlockMPO`](@ref).

`ITensorMPS` does produce the Jordan form the paper's constructions need,
but in the reversed index convention: its virtual index `1` is the
paper's finished level `(3)` and its index `end` is the paper's starting
level `(1)`, making the tensor lower rather than upper triangular. This
function reverses the virtual indices so the result is in the paper's
ordering, and labels the middle block with `channel`, which becomes the
subscript `a` of the `2ₐ`/`3ₐ` levels under the rewiring of Eq. (63).

The edge tensors of a finite MPO carry only one virtual leg; they are
placed into row `1` and column `end` respectively, which is where the
boundary vectors select them.
"""
function block_mpo(H::MPO; channel::Integer = 1)
    n = length(H)
    n >= 2 || throw(ArgumentError("`block_mpo` needs at least two sites, got $n."))
    sites = collect(firstsiteinds(H))
    links = [linkind(H, j) for j in 1:(n - 1)]
    d = dim(links[1])
    all(dim(l) == d for l in links) ||
        throw(ArgumentError("`block_mpo` currently requires a uniform link dimension, got $(dim.(links))."))

    # Paper index p corresponds to ITensors index d + 1 - p.
    rev(i) = d + 1 - i

    W = [Dict{Tuple{Int, Int}, ITensor}() for _ in 1:n]
    for j in 1:n
        T = H[j]
        ll = j == 1 ? nothing : links[j - 1]
        rl = j == n ? nothing : links[j]
        rows = isnothing(ll) ? (1:1) : (1:d)
        cols = isnothing(rl) ? (1:1) : (1:d)
        for a in rows, b in cols
            blk = T
            isnothing(ll) || (blk = blk * onehot(ll => a))
            isnothing(rl) || (blk = blk * onehot(rl => b))
            norm(blk) > 1.0e-14 || continue
            # An absent leg means the tensor is already the boundary
            # row (site 1) or column (site n).
            r = isnothing(ll) ? 1 : rev(a)
            c = isnothing(rl) ? d : rev(b)
            W[j][(r, c)] = blk
        end
    end

    # The edge tensors of a finite MPO come pre-contracted with the
    # first-degree boundary vectors, which discards entries those vectors
    # cannot reach. The diagonal identities are the only such entries the
    # extensive construction later needs: rerouting the finished levels
    # moves the exit from level `(3)` to level `(1)`, and the last site's
    # `(1,1)` identity — the "no term in progress here" path — is exactly
    # what the old exit could not see. Restoring them is invisible to the
    # first-degree reading, whose boundary vectors still skip them.
    # Both identities are already present in the opposite corner of the
    # same tensor, so they are copied rather than rebuilt — that keeps
    # the QN arrows correct for free.
    last_id = _getop(W[n], d, d)
    first_id = _getop(W[1], 1, 1)
    (isnothing(last_id) || isnothing(first_id)) &&
        throw(ArgumentError("the MPO edge tensors carry no identity on the diagonal; `block_mpo` expects an `OpSum`-built Hamiltonian."))
    _addop!(W[n], 1, 1, last_id)
    _addop!(W[1], d, d, first_id)

    levels = Level[[LEVEL_ONE]]
    for _ in 2:(d - 1)
        push!(levels, [LevelTag(2, channel)])
    end
    push!(levels, [LevelTag(3, channel)])

    return BlockMPO(sites, W, levels)
end

"""
    to_mpo(B::BlockMPO)

Contract a [`BlockMPO`](@ref) back into an ordinary `ITensorMPS` MPO by
materializing its virtual indices, taking row `B.vL` on the first site
and column `B.vR` on the last.

For a first-degree `B` this reproduces the Hamiltonian it encodes, which
makes it the round-trip check against [`block_mpo`](@ref). For the
extensive MPOs produced by the Dyson construction it is how the result
becomes applicable to an MPS.
"""
function to_mpo(B::BlockMPO)
    n = length(B)
    d = virtualdim(B)
    links = [Index(d, "Link,l=$j") for j in 1:(n - 1)]
    tensors = Vector{ITensor}(undef, n)
    for j in 1:n
        ll = j == 1 ? nothing : links[j - 1]
        rl = j == n ? nothing : links[j]
        T = nothing
        for ((r, c), op) in B.W[j]
            # The boundary vectors keep only row `vL` of the first site
            # and column `vR` of the last; the rest is unreachable.
            j == 1 && r != B.vL && continue
            j == n && c != B.vR && continue
            proj = op
            isnothing(ll) || (proj = proj * onehot(ll => r))
            isnothing(rl) || (proj = proj * onehot(rl => c))
            T = isnothing(T) ? proj : T + proj
        end
        isnothing(T) && throw(ArgumentError("site $j of the `BlockMPO` has no reachable entries."))
        tensors[j] = T
    end
    return MPO(tensors)
end
