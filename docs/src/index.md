```@meta
CurrentModule = ITensorTDMPO
```

# ITensorTDMPO.jl

An [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) extension
for time evolution under time-dependent Hamiltonians, in particular
adiabatic ramps.

## Overview

The Hamiltonian is always described the same way — as a set of **driving
channels** `H(t) = Σₐ fₐ(t) H⁽ᵃ⁾`, each a time-independent MPO paired with
a scalar driving function (or a plain number, for a constant coefficient)
— and a single entry point, [`time_evolve`](@ref), selects the integrator:

```julia
using ITensorMPS, ITensorTDMPO

ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                nsteps = 100, cutoff = 1e-10, maxdim = 128)

# Same Hamiltonian, different integrator — nothing else changes.
ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "magnus", nsteps = 100)
```

Four algorithms are available — see [The time_evolve interface](@ref) for
the full keyword reference and a guide to which one to use:

| `alg` | Treatment of `H(t)` within a step | Order | Unitary |
|---|---|---|---|
| `"piecewise_constant"` | frozen at one evaluation point | 2 | yes |
| `"dyson"` | expanded to order `N` in the Dyson series | `N` | approximately |
| `"magnus"` | `Ω₁+Ω₂(+Ω₃)`, applies `exp(Ω)` | ~4 | yes |
| `"cfet"` *(default)* | product of exponentials, no commutators | 4 | yes |

The Dyson and Magnus constructions implement (part of) the algorithms of
[Vanthilt, Van Damme, Haegeman, McCulloch & Vanderstraeten,
*Matrix Product Operator Encodings of the Magnus Expansion and Dyson
Series*](https://arxiv.org/abs/2605.21597) — see [Scope and limitations](@ref)
for what is and is not implemented relative to the paper.

## Installation

Registered in the Julia General registry. From the Julia REPL:

```julia
using Pkg
Pkg.add("ITensorTDMPO")
```

## Citation

If you use the Dyson or Magnus constructions, please cite the paper this
package implements:

```bibtex
@article{Vanthilt2025,
  title   = {Matrix Product Operator Encodings of the Magnus Expansion and Dyson Series},
  author  = {Vanthilt, Victor and Van Damme, Maarten and Haegeman, Jutho and McCulloch, Ian P. and Vanderstraeten, Laurens},
  journal = {arXiv preprint arXiv:2605.21597},
  year    = {2025},
}
```
