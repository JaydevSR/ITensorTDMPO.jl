module TDVPlus

using ITensors
using ITensors: QN, hasqns, op
using ITensorMPS: ITensorMPS, MPO, MPS, apply, expand, firstsiteinds, inner, linkind,
    maxlinkdim, siteinds, tdvp, truncate
using LinearAlgebra: LinearAlgebra, norm

# The finite-state-machine layer the paper's constructions operate on.
# `DrivingChannels` and the drivers below do not depend on it yet.
include("fsm/levels.jl")
include("fsm/block_mpo.jl")

include("time_evolution/ramps.jl")
include("time_evolution/driven_hamiltonian.jl")
include("time_evolution/schedules.jl")
# `DrivingChannels` is the shared Hamiltonian representation, so it must be
# defined before the drivers that dispatch on it.
include("time_evolution/time_ordered_integrals.jl")
include("time_evolution/driving_channels.jl")
# The FSM Dyson construction needs both `DrivingChannels` and the
# time-ordered integrals, so it follows them.
include("fsm/dyson_fsm.jl")
include("fsm/compression.jl")
include("time_evolution/piecewise_tdvp.jl")
include("time_evolution/dyson.jl")
include("time_evolution/magnus.jl")
include("time_evolution/cfet.jl")
include("time_evolution/time_evolve.jl")
include("time_evolution/adaptive.jl")
include("observables/measures.jl")
include("observables/observer.jl")

export
    # Unified entry point
    time_evolve,
    EVOLUTION_ALGORITHMS,
    # Adaptive stepping
    adaptive_time_evolve,
    trace_distance,
    # Commutator-free propagator
    cfet_evolve,
    cfet_exponents,
    # Observables and diagnostics
    EvolutionObserver,
    observe!,
    results,
    entanglement_entropy,
    entanglement_profile,
    instantaneous_energy,
    energy_variance,
    instantaneous_spectrum,
    instantaneous_gap,
    adiabatic_report,
    # Ramp shapes and ramps
    RampShape,
    LinearRamp,
    SmoothstepRamp,
    SmootherstepRamp,
    SineRamp,
    SineSquaredRamp,
    PowerLawRamp,
    ExponentialRamp,
    Ramp,
    # Piecewise-constant TDVP driver
    DrivenHamiltonian,
    TDVPStepSpec,
    default_tdvp_schedule,
    piecewise_constant_tdvp,
    # Driving-channel decomposition
    DrivingChannels,
    nchannels,
    identity_mpo,
    commutator,
    # Time-ordered integrals
    cumulative_integral,
    time_ordered_integral,
    # Dyson series
    dyson_terms,
    dyson_mpo,
    dyson_evolve,
    # Magnus expansion
    magnus_terms,
    magnus_generator,
    magnus_evolve

end # module TDVPlus
