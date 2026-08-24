module TDVPlus

using ITensors
using ITensorMPS: ITensorMPS, MPO, MPS, apply, expand, firstsiteinds, inner, maxlinkdim,
    siteinds, tdvp
using LinearAlgebra: LinearAlgebra, norm

include("time_evolution/ramps.jl")
include("time_evolution/driven_hamiltonian.jl")
include("time_evolution/schedules.jl")
# `DrivingChannels` is the shared Hamiltonian representation, so it must be
# defined before the drivers that dispatch on it.
include("time_evolution/time_ordered_integrals.jl")
include("time_evolution/driving_channels.jl")
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
