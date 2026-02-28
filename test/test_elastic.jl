using Test
using JuliWave

@testset "Elastic Forward Solver" begin
    # Small homogeneous model
    nx, ny = 101, 101
    dx, dy = 10.0, 10.0
    grid = Grid2D(nx, ny, dx, dy)

    vp_val = 3300.0
    vs_val = vp_val / 1.732
    rho_val = 2800.0
    vp = fill(vp_val, nx, ny)
    vs = fill(vs_val, nx, ny)
    rho = fill(rho_val, nx, ny)
    model = ElasticModel2D(vp, vs, rho, grid)

    # Source at center with 45 degree angle
    f0 = 7.0
    wavelet = GaussianDerivativeWavelet(f0; amplitude=1e7)
    src = PointSource(500.0, 500.0, wavelet; angle=45.0)
    rec1 = Receiver(200.0, 500.0)
    rec2 = Receiver(800.0, 500.0)
    geometry = Geometry([src], [rec1, rec2])

    dt = suggest_dt(vp_val, dx, dy; courant_target=0.4)
    nt = 200
    config = SimulationConfig(nt, dt; pml_points=10)

    @testset "Basic simulation runs" begin
        seis_vx, seis_vy = simulate_elastic(model, geometry, config)
        @test size(seis_vx) == (nt, 2)
        @test size(seis_vy) == (nt, 2)
        @test !any(isnan, seis_vx)
        @test !any(isnan, seis_vy)
        @test !any(isinf, seis_vx)
        @test !any(isinf, seis_vy)
    end

    @testset "Stability" begin
        seis_vx, seis_vy = simulate_elastic(model, geometry, config)
        @test maximum(abs.(seis_vx)) < 1e25
        @test maximum(abs.(seis_vy)) < 1e25
    end

    @testset "Non-zero output for angled source" begin
        seis_vx, seis_vy = simulate_elastic(model, geometry, config)
        # With 45 degree angle, both components should have signal
        @test maximum(abs.(seis_vx)) > 0.0
        @test maximum(abs.(seis_vy)) > 0.0
    end

    @testset "Wavefield simulation" begin
        seis_vx, seis_vy, snaps_vx, snaps_vy = simulate_elastic_wavefield(
            model, geometry, config; save_every=50)
        @test size(snaps_vx, 1) == nx
        @test size(snaps_vx, 2) == ny
        @test !any(isnan, snaps_vx)
        @test !any(isnan, snaps_vy)
    end
end
