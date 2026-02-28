"""
    simulate_elastic(model, geometry, config; backend=Array) -> (seis_vx, seis_vy)

Run a full 2D elastic forward simulation with C-PML absorbing boundaries.
Returns seismograms as two matrices (nt × nrec) for vx and vy components.

Translated from seismic_CPML_2D_isotropic_second_order.f90
"""
function simulate_elastic(model::ElasticModel2D, geometry::Geometry,
                          config::SimulationConfig; backend=Array)
    grid = model.grid
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    nt = config.nt

    # Check CFL condition
    vp_max = maximum(model.vp)
    check_cfl(vp_max, dt, dx, dy)

    # Get dominant frequency from first source
    f0 = geometry.sources[1].wavelet.f0

    # Setup CPML
    cpml = setup_cpml(grid, config, vp_max, f0; backend=backend)

    # Compute Lame parameters: lambda = rho * (vp² - 2*vs²), mu = rho * vs²
    lambda = model.rho .* (model.vp .^ 2 .- 2.0 .* model.vs .^ 2)
    mu = model.rho .* model.vs .^ 2

    # Initialize state
    state = ElasticState2D(nx, ny; backend=backend)

    # Snap receivers to grid
    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    seismograms_vx = zeros(Float64, nt, nrec)
    seismograms_vy = zeros(Float64, nt, nrec)

    # Process each source
    for src in geometry.sources
        # Reset state
        state.vx .= 0.0
        state.vy .= 0.0
        state.sigma_xx .= 0.0
        state.sigma_yy .= 0.0
        state.sigma_xy .= 0.0
        state.mem_dvx_dx .= 0.0
        state.mem_dvx_dy .= 0.0
        state.mem_dvy_dx .= 0.0
        state.mem_dvy_dy .= 0.0
        state.mem_dsxx_dx .= 0.0
        state.mem_dsyy_dy .= 0.0
        state.mem_dsxy_dx .= 0.0
        state.mem_dsxy_dy .= 0.0

        # Source grid position
        isrc, jsrc = snap_to_grid(src.x, src.y, grid)

        # Precompute source time series
        source_ts = compute_source_timeseries(src.wavelet, nt, dt)

        # Force angle decomposition
        angle_rad = src.angle * π / 180.0

        # Time stepping
        for it in 1:nt
            # Stress update from velocity gradients
            elastic_stress_update!(state, lambda, mu, cpml, nx, ny, dx, dy, dt)

            # Velocity update from stress divergence
            elastic_velocity_update!(state, model.rho, cpml, nx, ny, dx, dy, dt)

            # Source injection
            force_x = sin(angle_rad) * source_ts[it]
            force_y = cos(angle_rad) * source_ts[it]
            elastic_apply_source!(state, model.rho, force_x, force_y, isrc, jsrc, dt)

            # Dirichlet boundary conditions
            elastic_apply_bc!(state, nx, ny)

            # Record receivers
            record_receivers!(seismograms_vx, state.vx, rec_indices, it)
            record_receivers!(seismograms_vy, state.vy, rec_indices, it)
        end
    end

    return seismograms_vx, seismograms_vy
end

"""
    simulate_elastic_wavefield(model, geometry, config; save_every=1, backend=Array)

Run elastic simulation and return seismograms plus wavefield snapshots.
Returns (seis_vx, seis_vy, snaps_vx, snaps_vy).
"""
function simulate_elastic_wavefield(model::ElasticModel2D, geometry::Geometry,
                                    config::SimulationConfig;
                                    save_every::Int=1, backend=Array)
    grid = model.grid
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    nt = config.nt

    vp_max = maximum(model.vp)
    check_cfl(vp_max, dt, dx, dy)

    f0 = geometry.sources[1].wavelet.f0
    cpml = setup_cpml(grid, config, vp_max, f0; backend=backend)
    lambda = model.rho .* (model.vp .^ 2 .- 2.0 .* model.vs .^ 2)
    mu = model.rho .* model.vs .^ 2
    state = ElasticState2D(nx, ny; backend=backend)

    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    seismograms_vx = zeros(Float64, nt, nrec)
    seismograms_vy = zeros(Float64, nt, nrec)

    n_snaps = length(1:save_every:nt)
    snaps_vx = zeros(Float64, nx, ny, n_snaps)
    snaps_vy = zeros(Float64, nx, ny, n_snaps)
    snap_idx = 0

    src = geometry.sources[1]
    isrc, jsrc = snap_to_grid(src.x, src.y, grid)
    source_ts = compute_source_timeseries(src.wavelet, nt, dt)
    angle_rad = src.angle * π / 180.0

    for it in 1:nt
        elastic_stress_update!(state, lambda, mu, cpml, nx, ny, dx, dy, dt)
        elastic_velocity_update!(state, model.rho, cpml, nx, ny, dx, dy, dt)

        force_x = sin(angle_rad) * source_ts[it]
        force_y = cos(angle_rad) * source_ts[it]
        elastic_apply_source!(state, model.rho, force_x, force_y, isrc, jsrc, dt)
        elastic_apply_bc!(state, nx, ny)

        record_receivers!(seismograms_vx, state.vx, rec_indices, it)
        record_receivers!(seismograms_vy, state.vy, rec_indices, it)

        if mod(it - 1, save_every) == 0
            snap_idx += 1
            snaps_vx[:, :, snap_idx] .= state.vx
            snaps_vy[:, :, snap_idx] .= state.vy
        end
    end

    return seismograms_vx, seismograms_vy, snaps_vx, snaps_vy
end
