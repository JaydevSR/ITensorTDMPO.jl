# ITensorTDMPO.jl

[![CI](https://github.com/JaydevSR/ITensorTDMPO.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JaydevSR/ITensorTDMPO.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/JaydevSR/ITensorTDMPO.jl/actions/workflows/Documentation.yml/badge.svg)](https://jaydevsr.github.io/ITensorTDMPO.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

ITensorTDMPO.jl provides time evolution for time-dependent Hamiltonians on
top of [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) — in
particular, adiabatic ramps.

The Hamiltonian is described once, as a set of driving channels `H(t) =
Σₐ fₐ(t) H⁽ᵃ⁾`, and a single entry point, `time_evolve`, selects among
several integrators (piecewise-constant TDVP, the Dyson series, the
Magnus expansion, and a commutator-free exponential propagator), plus
adaptive step-size control, imaginary time, and adiabaticity
diagnostics. The Dyson and Magnus constructions implement part of the
algorithms of
[Vanthilt, Van Damme, Haegeman, McCulloch & Vanderstraeten (2025)](https://arxiv.org/abs/2605.21597).

Not registered yet; not affiliated with the ITensor project.

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/ITensorTDMPO.jl")
```

## Documentation

The full documentation — every algorithm, benchmark, and the API
reference — is at **[jaydevsr.github.io/ITensorTDMPO.jl](https://jaydevsr.github.io/ITensorTDMPO.jl/dev/)**.

To build it locally instead:

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); include("docs/make.jl")'
```

then open `docs/build/index.html`.

## Quick example

```julia
using ITensorMPS, ITensorTDMPO

ramp = Ramp(SmoothstepRamp(), 0.0, 10.0, 0.0, 2.0)

ψ = time_evolve([(1.0, Hzz), (ramp, Hx)], ψ0, 0.0, 10.0;
                alg = "cfet", nsteps = 100, cutoff = 1e-10, maxdim = 128)
```

## Testing

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Citation

If you use the Dyson or Magnus constructions, please cite the paper
this package implements:

```bibtex
@article{Vanthilt2025,
  title   = {Matrix Product Operator Encodings of the Magnus Expansion and Dyson Series},
  author  = {Vanthilt, Victor and Van Damme, Maarten and Haegeman, Jutho and McCulloch, Ian P. and Vanderstraeten, Laurens},
  journal = {arXiv preprint arXiv:2605.21597},
  year    = {2025},
}
```
