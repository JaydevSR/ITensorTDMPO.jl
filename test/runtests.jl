using TDVPlus
using Test

@testset "TDVPlus.jl" begin
    include("test_ramps.jl")
    include("test_piecewise_tdvp.jl")
    include("test_time_ordered_integrals.jl")
    include("test_driving_channels.jl")
    include("test_dyson_magnus.jl")
    include("test_time_evolve.jl")
    include("test_cfet_adaptive.jl")
    include("test_observables.jl")
    include("test_long_range.jl")
end
