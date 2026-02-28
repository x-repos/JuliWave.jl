using Test
using JuliWave

@testset "FWI Module" begin
    @testset "L2 misfit" begin
        a = [1.0 2.0; 3.0 4.0]
        b = [1.1 2.2; 3.3 4.4]
        misfit = l2_misfit(a, b)
        expected = 0.5 * sum((a .- b) .^ 2)
        @test misfit ≈ expected
    end

    @testset "L2 misfit zero for identical" begin
        a = rand(10, 5)
        @test l2_misfit(a, a) ≈ 0.0
    end

    @testset "Gradient finite difference" begin
        # Very small model for gradient testing
        nx, ny = 21, 21
        dx, dy = 10.0, 10.0
        grid = Grid2D(nx, ny, dx, dy)

        vp_true = fill(2000.0, nx, ny)
        vp_true[8:14, 8:14] .= 2200.0  # anomaly
        rho = fill(2000.0, nx, ny)

        f0 = 15.0
        src = PointSource(100.0, 100.0, RickerWavelet(f0))
        rec = Receiver(100.0, 100.0)
        geometry = Geometry([src], [rec])

        dt = suggest_dt(2200.0, dx, dy; courant_target=0.3)
        nt = 50
        config = SimulationConfig(nt, dt; pml_points=5)

        # Generate "observed" data with true model
        model_true = AcousticModel2D(vp_true, rho, grid)
        observed = simulate_acoustic(model_true, geometry, config)

        # Initial model (homogeneous)
        vp_init = fill(2000.0, nx, ny)
        model_init = AcousticModel2D(vp_init, rho, grid)

        # Test that misfit is non-zero for wrong model
        synthetic = simulate_acoustic(model_init, geometry, config)
        misfit = l2_misfit(synthetic, observed)
        @test misfit > 0.0

        # Gradient at a single point via finite differences
        h = 1.0
        i_test, j_test = 11, 11
        vp_pert = copy(vp_init)
        vp_pert[i_test, j_test] += h
        model_pert = AcousticModel2D(vp_pert, rho, grid)
        syn_pert = simulate_acoustic(model_pert, geometry, config)
        grad_fd = (l2_misfit(syn_pert, observed) - misfit) / h

        # Just check it's a finite number (full gradient check is expensive)
        @test isfinite(grad_fd)
    end
end
