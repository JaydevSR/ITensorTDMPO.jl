```@meta
CurrentModule = ITensorTDMPO
```

# Scope and limitations

**What is implemented:** the paper's physical structure — the
driving-channel decomposition, the time-ordered integrals, and the Dyson
and Magnus expansions expressed in that language — *and* its central
technical contribution, the size-extensive finite-state-machine MPO
encoding of Sec. V–VI: the `{L, R, A, D}` first-degree block structure,
the rewiring that gives every driving channel its own finished level
`3ₐ`, the `N`-th order Dyson construction of Algorithm 2, and the exact
equivalent-column compression of Sec. VI A. [`dyson_evolve`](@ref) and
[`dyson_mpo_fsm`](@ref) use this construction by default.

The *direct* construction — operator strings formed by explicit MPO
multiplication (`apply(A, B)`) rather than by manipulating the FSM block
structure — remains implemented and exported as [`dyson_mpo`](@ref) and
[`dyson_terms`](@ref). It is not size-extensive, and its accuracy
degrades with chain length; see the measured comparison below. It is
kept because it is useful as an independent reference to verify the FSM
construction against, and for direct inspection of individual Dyson
terms.

**What is not implemented:** the paper's *approximate row compression*
(Sec. VI B). This is a deliberate scope decision, not an oversight — see
[Why row compression is not implemented](@ref) below.

The practical consequences:

- **Finite systems only.** `ITensorMPS` represents only finite MPS/MPO,
  so everything here is finite by construction; infinite systems would
  need `ITensorInfiniteMPS.jl` or MPSKit.jl.
- **[`dyson_evolve`](@ref) is size-extensive**: its per-site error does
  not grow with chain length. Measured on an Ising chain (`Sz·Sz` +
  transverse field, two driving channels, order 2), the per-site
  infidelity after a fixed evolution time changes by less than a factor
  of 2 from a 4-site to an 8-site chain, while the *direct* construction
  of [`dyson_mpo`](@ref)/[`dyson_terms`](@ref) grows by a factor of
  order 10 over the same range, matching its `N^4.7`-ish scaling. If you
  need the direct construction's accuracy on more than a handful of
  sites, prefer [`dyson_evolve`](@ref) over building steps from
  [`dyson_mpo`](@ref) yourself.
- **Bond dimension is competitive with the paper's compressed
  construction on finite systems**, once truncation is accounted for —
  see the next section for the numbers, and why that makes row
  compression unnecessary here.
- **Cost is exponential in the order for construction**, before
  truncation. The order-`N` Dyson block MPO is built as an `N`-fold
  product over the rewired Hamiltonian's virtual space, so its
  *untruncated* bond dimension grows combinatorially in `N` and in the
  number of channels — fine for `order ≤ 3` and a few channels, the same
  regime the direct construction was already limited to.

## Why row compression is not implemented

The paper's Sec. VI B algebraically compresses the Dyson MPO's bond
dimension without ever forming the (larger) uncompressed operator
explicitly — which matters for an infinite system, where there is no
finite bond to run an SVD over and no other way to keep the bond
dimension in check.

`ITensorMPS`, and therefore this package, only represents finite
systems. On a finite chain, [`dyson_mpo_fsm`](@ref) can simply truncate
the assembled MPO with an ordinary SVD after Algorithm 2 and the exact
column compression — and at a tight enough cutoff, this reaches bond
dimensions **at or below the paper's own Table I** (its published
result *after* row compression), for free, with the convergence order
unaffected. Measured on an XXZ chain (`χ = 3` per channel):

| model | order | column-compressed | after SVD (`cutoff = 1e-14`) | paper Table I |
|---|---|---|---|---|
| `Sz·Sz`, `χ = 1` | 3 | 12 | 5 | 5 |
| XXZ, `χ = 3` | 2 | 19 | 8 | 13 |
| XXZ, `χ = 3` | 3 | 82 | 14 | 43 |

and the convergence rate is unaffected by the truncation at this
cutoff — confirmed down to the point where the untruncated and truncated
errors agree to three significant figures. (A looser cutoff can silently
cap the achievable order; see the warning on
[`dyson_mpo_fsm`](@ref).)

Implementing the row compression algorithm on top of this would not
improve the final MPO this package ships — SVD already reaches or beats
it. Its only remaining value here would be reducing the *intermediate*
memory of the `N`-fold product Algorithm 2 builds before column
compression and truncation get a chance to run, which matters only at
higher orders or channel counts than this package currently targets. If
that becomes a real bottleneck, row compression (or an equivalent
numerical rank-revealing approach, which does not require reproducing
the paper's symbolic level enumeration) is the natural next step —
tracked as future work rather than a gap in the current implementation.

## Long-range interactions

The paper's constructions handle long-range (e.g. exponentially
decaying) interactions without modification, since `H⁽ᵃ⁾` channels can
be arbitrary MPOs — nothing in either construction used here assumes
short range. See the test suite for a verified example with
exponentially-decaying long-range coupling.

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
evolution with flux preservation (for both the direct and FSM
constructions), the time-ordered-integral closed forms and the factoring
property, that every `time_evolve` algorithm reproduces its direct
driver exactly, the CFET weights and its fourth-order convergence,
adaptive stepping against its own tolerance, imaginary time converging
to the DMRG ground-state energy, long-range interactions, and the
observable/diagnostic layer (entropy of a Bell pair, zero variance on an
eigenstate, gap against a direct DMRG solve).

The FSM construction specifically is checked by: an exact round-trip
through the internal first-degree block extraction and reconstruction
for several Hamiltonian shapes (nearest-neighbour, long-range, and XXZ,
which has a `χ > 1` middle block); the `O(dtᴺ⁺¹)` convergence of the
order-`N` construction against a dense exponential; the size-extensivity
comparison above, against the direct construction on the same model; the
exact-to-machine-precision correctness of the column compression,
verified by comparing the compressed and uncompressed operators directly
rather than only their accuracy against a reference (which would not
catch a compression that merges non-equivalent levels but happens to
stay within tolerance); and quantum-number conservation, including a
regression test on a `χ > 1` block where two levels are equivalent in
label but carry different flux and must not be merged.

Note that these drivers do not renormalize, so the state norm drifts by
~1e-11 under truncation. Compare states with a normalized overlap
`abs(inner(a, b)) / (norm(a) * norm(b))`, not with `abs(inner(a, b))`
against 1.
