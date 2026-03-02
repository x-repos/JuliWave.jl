using Test
using JuliWave

@testset "Acoustic Forward Solver" begin
    # Small homogeneous model for testing
    nx, ny = 101, 101
    dx, dy = 10.0, 10.0
    grid = Grid2D(nx, ny, dx, dy)

    vp_val = 2000.0
    rho_val = 2000.0
    vp = fill(vp_val, nx, ny)
    rho = fill(rho_val, nx, ny)
    model = AcousticModel2D(vp, rho, grid)

    # Source at center
    f0 = 15.0
    src = PointSource(500.0, 500.0, RickerWavelet(f0))
    rec1 = Receiver(200.0, 500.0)
    rec2 = Receiver(800.0, 500.0)
    geometry = Geometry([src], [rec1, rec2])

    dt = suggest_dt(vp_val, dx, dy; courant_target=0.4)
    nt = 200
    config = SimulationConfig(nt, dt; pml_points=10)

    @testset "Basic simulation runs" begin
        seismograms = simulate_acoustic(model, geometry, config)
        @test size(seismograms) == (nt, 2)
        @test !any(isnan, seismograms)
        @test !any(isinf, seismograms)
    end

    @testset "Stability" begin
        seismograms = simulate_acoustic(model, geometry, config)
        @test maximum(abs.(seismograms)) < 1e25
    end

    @testset "Symmetric receivers see similar signal" begin
        # Source at center, two receivers equidistant -> similar amplitudes
        seismograms = simulate_acoustic(model, geometry, config)
        max1 = maximum(abs.(seismograms[:, 1]))
        max2 = maximum(abs.(seismograms[:, 2]))
        # They should be roughly equal (same distance from source)
        @test abs(max1 - max2) / max(max1, max2) < 0.1
    end

    @testset "CFL check" begin
        @test_throws ErrorException SimulationConfig(100, 1.0; pml_points=10) |>
            cfg -> simulate_acoustic(model, geometry, cfg)
    end

    @testset "Wavefield simulation" begin
        seismograms, snapshots = simulate_acoustic_wavefield(model, geometry, config; save_every=50)
        @test size(snapshots, 1) == nx
        @test size(snapshots, 2) == ny
        @test size(snapshots, 3) == length(1:50:nt)
        @test !any(isnan, snapshots)
    end

    @testset "Free surface" begin
        config_fs = SimulationConfig(nt, dt; pml_points=10, free_surface=true)

        @testset "Basic simulation runs" begin
            seismograms = simulate_acoustic(model, geometry, config_fs)
            @test size(seismograms) == (nt, 2)
            @test !any(isnan, seismograms)
            @test !any(isinf, seismograms)
            @test maximum(abs.(seismograms)) < 1e25
        end

        @testset "Wavefield simulation" begin
            seismograms, snapshots = simulate_acoustic_wavefield(model, geometry, config_fs; save_every=50)
            @test size(snapshots, 1) == nx
            @test size(snapshots, 2) == ny
            @test size(snapshots, 3) == length(1:50:nt)
            @test !any(isnan, snapshots)
        end
    end

    @testset "Higher-order FD (space_order=$order)" for order in [4, 8]
        dt_ho = suggest_dt(vp_val, dx, dy; courant_target=0.4, space_order=order)
        config_ho = SimulationConfig(nt, dt_ho; pml_points=10, space_order=order)

        seismograms = simulate_acoustic(model, geometry, config_ho)
        @test size(seismograms) == (nt, 2)
        @test !any(isnan, seismograms)
        @test !any(isinf, seismograms)
        @test maximum(abs.(seismograms)) < 1e25

        seis_wf, snaps = simulate_acoustic_wavefield(model, geometry, config_ho; save_every=50)
        @test size(snaps, 1) == nx
        @test size(snaps, 2) == ny
        @test !any(isnan, snaps)
    end
end
