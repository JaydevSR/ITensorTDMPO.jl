"""
    equivalent_column_groups(B::BlockMPO)

Group `B`'s virtual levels by operator history, as the exact compression
of the paper's Sec. VI A prescribes: two levels are equivalent when their
labels agree after every `1` is stripped out.

A `1` in a label marks a factor that has not started yet, so sliding it
through the label cannot change which operators came before. The levels
`(1 2₁ 3₂)`, `(2₁ 1 3₂)` and `(2₁ 3₂ 1)` therefore describe the same
history and are interchangeable.

Returns the groups with more than one member, each as a vector of level
indices in increasing order.
"""
function equivalent_column_groups(B::BlockMPO)
    groups = Dict{Level, Vector{Int}}()
    for (i, l) in enumerate(B.levels)
        push!(get!(groups, strip_ones(l), Int[]), i)
    end
    return [sort(g) for g in values(groups) if length(g) > 1]
end

"""
    compress_columns(B::BlockMPO; atol = 1e-10)

Merge equivalent columns of `B`, the *exact* compression step of the
paper's Sec. VI A. The operator `B` encodes is unchanged; only its bond
dimension shrinks.

For each group of levels sharing a history, this applies the gauge
transform of the paper's footnote to every site tensor, `W ↦ B W A`,
where `B` sums each group's rows into its surviving level and `A`
subtracts the surviving column from the others.

Note that equivalent levels do *not* have literally equal columns: the
same operator can arrive from different incoming levels, so the entries
sit in different rows. What makes them equivalent is that those incoming
levels are themselves an equivalent group, so the row sum brings the
entries into a common row before the column subtraction cancels them.
The two passes below are therefore over all groups at once, not group by
group.

Afterwards the discarded columns vanish on every *surviving* row, which
is the condition that matters: with no kept level transitioning into
them, they cannot be reached from the boundary, and whatever remains in
the discarded rows is unreachable bookkeeping. That is checked rather
than assumed — a residual above `atol` means the levels grouped by
history were not genuinely equivalent.
"""
function compress_columns(B::BlockMPO; atol = 1.0e-10)
    groups = equivalent_column_groups(B)
    isempty(groups) && return B

    drop = Int[]
    for g in groups
        append!(drop, g[2:end])
    end
    dropset = Set(drop)

    W = [copy(w) for w in B.W]

    # Pass 1 — the row sums of `B`. The snapshot is taken once per site,
    # before any group is folded in, so that a row is never counted
    # twice; the groups partition the levels, so they cannot interfere.
    for n in eachindex(W)
        snapshot = collect(pairs(W[n]))
        for g in groups
            keep = first(g)
            for l in g[2:end], ((r, c), op) in snapshot
                r == l || continue
                _addop!(W[n], keep, c, op)
            end
        end
    end

    # Pass 2 — the column subtractions of `A`, on the row-summed tensor.
    for n in eachindex(W)
        for g in groups
            keep = first(g)
            for l in g[2:end], r in 1:virtualdim(B)
                kept_op = _getop(W[n], r, keep)
                isnothing(kept_op) && continue
                _addop!(W[n], r, l, -kept_op)
            end
        end
    end

    for n in eachindex(W)
        for ((r, c), op) in W[n]
            (c in dropset && !(r in dropset)) || continue
            if norm(op) > atol
                throw(
                    ArgumentError(
                        "column $c of site $n still has norm $(norm(op)) on surviving row $r " *
                            "after equivalent-column compression; the levels grouped by " *
                            "history were not actually equivalent."
                    )
                )
            end
        end
    end

    return _drop_levels(BlockMPO(B.sites, W, B.levels, B.vL, B.vR), drop)
end
