using ITensorMPSExtended
using Test

@testset "ITensorMPSExtended.jl" begin
    include("test_ramps.jl")
    include("test_piecewise_tdvp.jl")
    include("test_time_ordered_integrals.jl")
    include("test_driving_channels.jl")
    include("test_dyson_magnus.jl")
end
