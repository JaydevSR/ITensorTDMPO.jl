"""
    LevelTag(kind, channel)

One slot of a finite-state-machine level label, in the `(1, 2, 3)`
notation of [Vanthilt et al.](https://arxiv.org/abs/2605.21597).

`kind` is `1` (the term has not started), `2` (a term is in progress), or
`3` (a term has finished). `channel` records the driving-channel
subscript `a` of the `2ₐ`/`3ₐ` levels introduced by the rewiring of
Eq. (63); it is `0` for `kind == 1`, which carries no subscript.
"""
struct LevelTag
    kind::Int8
    channel::Int8
end

LevelTag(kind::Integer, channel::Integer) = LevelTag(Int8(kind), Int8(channel))

const LEVEL_ONE = LevelTag(1, 0)

"""
    Level

A full FSM level label: the `N`-tuple of [`LevelTag`](@ref)s carried by a
virtual index of an `N`-fold product `H̃^N`. A first-degree MPO has
length-1 levels.
"""
const Level = Vector{LevelTag}

"""
    n_twos(l::Level)

The number of in-progress (`2ₐ`) slots in `l`, written `n₂` in the
paper's Algorithms 1 and 2.
"""
n_twos(l::Level) = count(t -> t.kind == 2, l)

"""
    n_threes(l::Level)

The number of finished (`3ₐ`) slots in `l`, written `n₃` in the paper's
Algorithms 1 and 2.
"""
n_threes(l::Level) = count(t -> t.kind == 3, l)

"""
    three_subscripts(l::Level)

The driving-channel subscripts of the `3ₐ` slots of `l`, in order —
the permutation `σ` of Algorithm 2, which selects the time-ordered
integral `[f_σ(1) ⋯ f_σ(n₃)]` that the level's column is rerouted with.
"""
three_subscripts(l::Level) = Int[t.channel for t in l if t.kind == 3]

"""
    strip_ones(l::Level)

`l` with all `1` slots removed. Two levels with the same stripped label
have identical operator histories and are merged by the exact
equivalent-column compression of the paper's Sec. VI A: a `1` marks a
factor that has not started, so moving it through the label does not
change the operators that came before.
"""
strip_ones(l::Level) = LevelTag[t for t in l if t.kind != 1]

"""
    is_start(l::Level)

Whether `l` is the all-`1`s level, i.e. the paper's level `(1)` — the
column that Algorithm 2 reroutes every finished level back into.
"""
is_start(l::Level) = all(t -> t.kind == 1, l)

function Base.show(io::IO, t::LevelTag)
    print(io, t.kind)
    return t.kind == 1 || print(io, "_", t.channel)
end

show_level(l::Level) = "(" * join(string.(l), " ") * ")"
