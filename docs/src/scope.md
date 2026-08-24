```@meta
CurrentModule = TDVPlus
```

# Scope and limitations

**What is implemented:** the paper's *physical structure* — the
driving-channel decomposition, the time-ordered integrals, and the
Dyson and Magnus expansions expressed in that language, to arbitrary
order (Dyson) and to third order (Magnus), plus a commutator-free (CFET)
integrator not from the paper.

**What is not:** the paper's central technical contribution, the
*size-extensive finite-state-machine MPO encoding*. Operator strings and
commutators here are formed by direct MPO multiplication (`apply(A,
B)`) rather than by manipulating the `{L, R, A, D}` block structure of
first-degree MPOs at the level of virtual "levels", and neither the
exact equivalent-column compression nor the approximate row compression
of the paper's Sec. VII is implemented. This is the natural next piece
of work — see the project's issue tracker / roadmap.

The practical consequences:

- **Finite systems only.** `ITensorMPS` represents only finite MPS/MPO,
  so everything here is finite by construction; infinite systems would
  need `ITensorInfiniteMPS.jl` or MPSKit.jl.
- **Accuracy degrades with chain length for `dyson_evolve`**, because
  the direct construction is not size-extensive — see the measured
  scaling on [The `time_evolve` interface](@ref) page.
  `magnus_evolve` is not affected, since exponentiating the generator
  resums the disjoint terms.
- **Bond dimension grows faster** than the paper's compressed
  construction. Intermediate products are truncated with
  `cutoff`/`maxdim` to keep this in check, but expect the MPO bond
  dimension to be the practical limit on order and system size.
- **Cost is exponential in the order.** The order-`N` Dyson MPO
  enumerates `nchannels^N` operator strings. Fine for `order ≤ 3` and a
  few channels.

Note that size-extensivity is **not** an infinite-system concern only.
The paper applies its encoding to finite systems too (Sec. VIII A
benchmarks a finite `L = 8` chain at exact bond dimension and recovers
clean `O(dtᴺ)` convergence). On an infinite system non-extensivity is
*fatal* — applying `Hⁿ` to a normalized uMPS gives a state that cannot
be normalized. On a finite system it is merely *harmful*: everything is
well defined, but the accuracy degrades with chain length, which is
exactly the `N^4.7` growth measured for `dyson_evolve`.

ITensors builds MPOs from `OpSum` with automatic compression and does
not expose the Jordan-block/first-degree structure that the paper's
algorithms operate on, so implementing the full encoding would mean
building that layer from scratch. That is the natural next step, and it
would fix the Dyson driver's chain-length scaling on finite systems — it
is worth doing even if you never want the thermodynamic limit.

## Long-range interactions

The paper's constructions handle long-range (e.g. exponentially
decaying) interactions without modification, since `H⁽ᵃ⁾` channels can
be arbitrary MPOs — nothing in the direct-multiplication construction
used here assumes short range either. See the test suite for a verified
example with exponentially-decaying long-range coupling.

## Testing

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The expansions are checked against **independent dense-matrix**
reference evolution on small chains, rather than against each other: the
reduction to the Taylor series for constant driving, anti-Hermiticity of
`Ω`, the equivalence of first-order Dyson and first-order Magnus noted
in the paper, the expected convergence order under step refinement, and
that both expansions beat freezing the Hamiltonian for a rapidly
oscillating drive. Also covered: QN-conserving (`conserve_qns = true`)
evolution with flux preservation, the time-ordered-integral closed forms
and the factoring property, that every `time_evolve` algorithm
reproduces its direct driver exactly, the CFET weights and its
fourth-order convergence, adaptive stepping against its own tolerance,
imaginary time converging to the DMRG ground-state energy, long-range
interactions, and the observable/diagnostic layer (entropy of a Bell
pair, zero variance on an eigenstate, gap against a direct DMRG solve).

Note that these drivers do not renormalize, so the state norm drifts by
~1e-11 under truncation. Compare states with a normalized overlap
`abs(inner(a, b)) / (norm(a) * norm(b))`, not with `abs(inner(a, b))`
against 1.
