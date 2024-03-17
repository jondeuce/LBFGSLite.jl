using LBFGSLite
using Test
using Aqua

@testset "LBFGSLite.jl" begin
    @testset "Basics" begin
        function fg!(x, dx)
            @. dx = sin(x) * cos(x)
            loss = 1 + sum(abs2 ∘ sin, x) / 2
            return loss
        end

        x₀ = Float64[-2.0, 1.0]
        params = LBFGSParams{Float64}()
        x, fx, st = optimize(fg!, x₀, params)

        @test x ≈ [-π, 0.0] atol = 1e-6
        @test fx ≈ 1.0 atol = 1e-12
        @test Int(st) == 0
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LBFGSLite)
    end
end
