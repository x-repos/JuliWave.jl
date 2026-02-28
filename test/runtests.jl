using Test
using JuliWave

@testset "JuliWave.jl" begin
    include("test_cpml.jl")
    include("test_acoustic.jl")
    include("test_elastic.jl")
    include("test_fwi.jl")
end
