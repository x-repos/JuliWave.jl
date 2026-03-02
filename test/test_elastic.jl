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

    @testset "Free surface" begin
        config_fs = SimulationConfig(nt, dt; pml_points=10, free_surface=true)

        @testset "Basic simulation runs" begin
            seis_vx, seis_vy = simulate_elastic(model, geometry, config_fs)
            @test size(seis_vx) == (nt, 2)
            @test size(seis_vy) == (nt, 2)
            @test !any(isnan, seis_vx)
            @test !any(isnan, seis_vy)
            @test !any(isinf, seis_vx)
            @test !any(isinf, seis_vy)
            @test maximum(abs.(seis_vx)) < 1e25
            @test maximum(abs.(seis_vy)) < 1e25
        end

        @testset "Wavefield simulation" begin
            seis_vx, seis_vy, snaps_vx, snaps_vy = simulate_elastic_wavefield(
                model, geometry, config_fs; save_every=50)
            @test size(snaps_vx, 1) == nx
            @test size(snaps_vx, 2) == ny
            @test !any(isnan, snaps_vx)
            @test !any(isnan, snaps_vy)
        end
    end

    @testset "Pressure source" begin
        # Pressure source at center
        src_p = PointSource(500.0, 500.0, wavelet)
        geometry_p = Geometry([src_p], [rec1, rec2])

        @testset "Basic simulation runs" begin
            seis_vx, seis_vy = simulate_elastic(model, geometry_p, config; src_type=:pressure)
            @test size(seis_vx) == (nt, 2)
            @test size(seis_vy) == (nt, 2)
            @test !any(isnan, seis_vx)
            @test !any(isnan, seis_vy)
            @test !any(isinf, seis_vx)
            @test !any(isinf, seis_vy)
            @test maximum(abs.(seis_vx)) < 1e25
            @test maximum(abs.(seis_vy)) < 1e25
        end

        @testset "Produces non-zero output" begin
            seis_vx, seis_vy = simulate_elastic(model, geometry_p, config; src_type=:pressure)
            @test maximum(abs.(seis_vx)) > 0.0
            @test maximum(abs.(seis_vy)) > 0.0
        end

        @testset "Isotropic radiation (symmetric receivers)" begin
            # Symmetric receivers equidistant from center source
            rec_left = Receiver(300.0, 500.0)
            rec_right = Receiver(700.0, 500.0)
            geom_sym = Geometry([src_p], [rec_left, rec_right])
            seis_vx, seis_vy = simulate_elastic(model, geom_sym, config; src_type=:pressure)
            # Pressure source is isotropic: symmetric receivers should see similar amplitudes
            max_left_vx = maximum(abs.(seis_vx[:, 1]))
            max_right_vx = maximum(abs.(seis_vx[:, 2]))
            # vx should be antisymmetric (opposite sign), so amplitudes should be similar
            # staggered grid introduces slight asymmetry, so use 15% tolerance
            @test isapprox(max_left_vx, max_right_vx, rtol=0.15)
        end

        @testset "Wavefield simulation" begin
            seis_vx, seis_vy, snaps_vx, snaps_vy = simulate_elastic_wavefield(
                model, geometry_p, config; save_every=50, src_type=:pressure)
            @test size(snaps_vx, 1) == nx
            @test size(snaps_vx, 2) == ny
            @test !any(isnan, snaps_vx)
            @test !any(isnan, snaps_vy)
        end

        @testset "With free surface" begin
            config_fs = SimulationConfig(nt, dt; pml_points=10, free_surface=true)
            seis_vx, seis_vy = simulate_elastic(model, geometry_p, config_fs; src_type=:pressure)
            @test !any(isnan, seis_vx)
            @test !any(isnan, seis_vy)
            @test !any(isinf, seis_vx)
            @test !any(isinf, seis_vy)
            @test maximum(abs.(seis_vx)) < 1e25
            @test maximum(abs.(seis_vy)) < 1e25
        end
    end

    @testset "Higher-order FD (space_order=$order)" for order in [4, 8]
        dt_ho = suggest_dt(vp_val, dx, dy; courant_target=0.4, space_order=order)
        config_ho = SimulationConfig(nt, dt_ho; pml_points=10, space_order=order)

        seis_vx, seis_vy = simulate_elastic(model, geometry, config_ho)
        @test size(seis_vx) == (nt, 2)
        @test size(seis_vy) == (nt, 2)
        @test !any(isnan, seis_vx)
        @test !any(isnan, seis_vy)
        @test !any(isinf, seis_vx)
        @test !any(isinf, seis_vy)
        @test maximum(abs.(seis_vx)) < 1e25
        @test maximum(abs.(seis_vy)) < 1e25

        svx, svy, snvx, snvy = simulate_elastic_wavefield(
            model, geometry, config_ho; save_every=50)
        @test size(snvx, 1) == nx
        @test size(snvx, 2) == ny
        @test !any(isnan, snvx)
        @test !any(isnan, snvy)
    end
end
