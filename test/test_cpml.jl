using Test
using JuliWave

@testset "CPML Coefficients" begin
    grid = Grid2D(101, 101, 10.0, 10.0)
    config = SimulationConfig(100, 1e-3; pml_points=10, pml_Rcoef=0.001, pml_npower=2.0)
    vmax = 3300.0
    f0 = 7.0

    cpml = setup_cpml(grid, config, vmax, f0)

    @testset "Dimensions" begin
        @test length(cpml.x.a) == grid.nx
        @test length(cpml.x.b) == grid.nx
        @test length(cpml.x.K) == grid.nx
        @test length(cpml.y.a) == grid.ny
        @test length(cpml.y.b) == grid.ny
        @test length(cpml.y.K) == grid.ny
    end

    @testset "Interior values" begin
        # In the interior (away from PML), coefficients should be trivial
        mid = div(grid.nx, 2)
        @test cpml.x.K[mid] ≈ 1.0
        @test cpml.x.a[mid] ≈ 0.0 atol=1e-10
        @test cpml.x.b[mid] ≈ 1.0 atol=1e-10  # exp(0) = 1
        @test cpml.y.K[mid] ≈ 1.0
        @test cpml.y.a[mid] ≈ 0.0 atol=1e-10
    end

    @testset "PML edge values" begin
        # At the boundary (i=1), damping should be maximal
        @test cpml.x.b[1] < 1.0  # b = exp(-something) < 1
        @test cpml.x.a[1] < 0.0  # a is negative (d*(b-1)/...)
        @test cpml.y.b[1] < 1.0
        @test cpml.y.a[1] < 0.0
    end

    @testset "Symmetry" begin
        # Left and right PML should be symmetric for equal grid
        @test cpml.x.b[1] ≈ cpml.x.b[end]
        @test cpml.x.a[1] ≈ cpml.x.a[end]
        @test cpml.x.K[1] ≈ cpml.x.K[end]
    end

    @testset "K values >= 1" begin
        @test all(cpml.x.K .>= 1.0)
        @test all(cpml.y.K .>= 1.0)
        @test all(cpml.x.K_half .>= 1.0)
        @test all(cpml.y.K_half .>= 1.0)
    end

    @testset "b values in [0,1]" begin
        @test all(0.0 .<= cpml.x.b .<= 1.0)
        @test all(0.0 .<= cpml.y.b .<= 1.0)
        @test all(0.0 .<= cpml.x.b_half .<= 1.0)
        @test all(0.0 .<= cpml.y.b_half .<= 1.0)
    end
end
