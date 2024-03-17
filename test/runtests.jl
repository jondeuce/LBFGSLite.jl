using LBFGSLite
using Test
using Aqua

@testset "LBFGSLite.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LBFGSLite)
    end
    # Write your tests here.
end
