"""
    TDVPStepSpec(; nsite, expand_krylov = false, expand_kwargs = (;))

Describes how a single piecewise-constant TDVP step should be carried
out: whether to use 1-site or 2-site updates (`nsite`), and whether to
perform a global Krylov subspace expansion (see `ITensorMPS.expand` with
`alg = "global_krylov"`) before the step. Subspace expansion is only
needed when `nsite == 1`, since 1-site TDVP cannot grow the MPS bond
dimension on its own.
"""
struct TDVPStepSpec{K <: NamedTuple}
    nsite::Int
    expand_krylov::Bool
    expand_kwargs::K
end

function TDVPStepSpec(; nsite, expand_krylov = false, expand_kwargs = (;))
    return TDVPStepSpec(nsite, expand_krylov, expand_kwargs)
end

"""
    default_tdvp_schedule(t, ψ)

Default TDVP calibration function: always use 2-site TDVP and never
perform a global Krylov expansion, since 2-site TDVP grows the bond
dimension on its own. `t` is the time at which the step is evaluated and
`ψ` is the current MPS state.
"""
default_tdvp_schedule(t, ψ) = TDVPStepSpec(; nsite = 2, expand_krylov = false)
