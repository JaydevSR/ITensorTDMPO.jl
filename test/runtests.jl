using TDVPlus
using Test

# `verbose = true` with one nested testset per file reports a per-file
# time breakdown, which is what tells you where a slow run went.
@testset verbose = true "TDVPlus.jl" begin
    for file in [
            "test_ramps.jl",
            "test_piecewise_tdvp.jl",
            "test_time_ordered_integrals.jl",
            "test_driving_channels.jl",
            "test_dyson_magnus.jl",
            "test_fsm_dyson.jl",
            "test_time_evolve.jl",
            "test_cfet_adaptive.jl",
            "test_observables.jl",
            "test_long_range.jl",
        ]
        @testset "$file" begin
            include(file)
        end
    end
end
