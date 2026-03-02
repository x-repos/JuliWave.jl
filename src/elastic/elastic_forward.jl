"""
    simulate_elastic(model, geometry, config; backend=Array, src_type=:force) -> (seis_vx, seis_vy)

Run a full 2D elastic forward simulation with C-PML absorbing boundaries.
Returns seismograms as two matrices (nt × nrec) for vx and vy components.

# Keywords
- `backend`: Array type (default `Array` for CPU)
- `src_type`: Source type — `:force` (default, directional force) or `:pressure` (explosive/pressure source injected into σ_xx, σ_yy)

Translated from seismic_CPML_2D_isotropic_second_order.f90
"""
function simulate_elastic(model::ElasticModel2D, geometry::Geometry,
                          config::SimulationConfig; backend=Array,
                          src_type::Symbol=:force)
    grid = model.grid
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    nt = config.nt

    # FD coefficients
    coeffs = fd_coefficients(config.space_order)

    # Check CFL condition
    vp_max = maximum(model.vp)
    check_cfl(vp_max, dt, dx, dy; space_order=config.space_order)

    # Get dominant frequency from first source
    f0 = geometry.sources[1].wavelet.f0

    # Setup CPML on original grid
    cpml = setup_cpml(grid, config, vp_max, f0; backend=backend)

    # Disable PML at the top boundary for free surface
    if config.free_surface
        _disable_cpml_left!(cpml.y, config.pml_points)
    end

    # Ghost cell padding for higher-order FD stencils
    pad = length(coeffs) - 1
    nxp, nyp = nx + 2 * pad, ny + 2 * pad

    if pad > 0
        vp_pad = pad_array(model.vp, pad)
        vs_pad = pad_array(model.vs, pad)
        rho_pad = pad_array(model.rho, pad)
        cpml = pad_cpml(cpml, pad)
    else
        vp_pad = model.vp
        vs_pad = model.vs
        rho_pad = model.rho
    end

    # Compute Lame parameters: lambda = rho * (vp² - 2*vs²), mu = rho * vs²
    lambda = rho_pad .* (vp_pad .^ 2 .- 2.0 .* vs_pad .^ 2)
    mu = rho_pad .* vs_pad .^ 2

    # Initialize state with padded dimensions
    state = ElasticState2D(nxp, nyp; backend=backend)

    # Snap receivers to grid (physical coordinates)
    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    rec_indices_pad = [(i + pad, j + pad) for (i, j) in rec_indices]
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

        # Source grid position (physical grid, then offset)
        isrc, jsrc = snap_to_grid(src.x, src.y, grid)
        isrc_pad, jsrc_pad = isrc + pad, jsrc + pad

        # Precompute source time series
        source_ts = compute_source_timeseries(src.wavelet, nt, dt)

        # Force angle decomposition
        angle_rad = src.angle * π / 180.0

        # Time stepping
        for it in 1:nt
            # Stress update from velocity gradients
            elastic_stress_update!(state, lambda, mu, cpml, nxp, nyp, dx, dy, dt, coeffs)

            # Free surface: zero σ_yy at top + mirror stresses into ghost cells
            if config.free_surface
                elastic_apply_free_surface!(state, nxp, nyp, pad)
            end

            # Pressure source injection (after stress update, before velocity update)
            if src_type == :pressure
                elastic_apply_pressure_source!(state, source_ts[it], isrc_pad, jsrc_pad, dt)
            end

            # Velocity update from stress divergence
            elastic_velocity_update!(state, rho_pad, cpml, nxp, nyp, dx, dy, dt, coeffs)

            # Force source injection (after velocity update)
            if src_type == :force
                force_x = sin(angle_rad) * source_ts[it]
                force_y = cos(angle_rad) * source_ts[it]
                elastic_apply_source!(state, rho_pad, force_x, force_y, isrc_pad, jsrc_pad, dt)
            end

            # Boundary conditions (free surface skips top edge)
            elastic_apply_bc!(state, nxp, nyp; pad=pad, free_surface=config.free_surface)

            # Record receivers
            record_receivers!(seismograms_vx, state.vx, rec_indices_pad, it)
            record_receivers!(seismograms_vy, state.vy, rec_indices_pad, it)
        end
    end

    return seismograms_vx, seismograms_vy
end

"""
    simulate_elastic_wavefield(model, geometry, config; save_every=1, backend=Array, src_type=:force)

Run elastic simulation and return seismograms plus wavefield snapshots.
Returns (seis_vx, seis_vy, snaps_vx, snaps_vy).

# Keywords
- `save_every`: Save wavefield every N time steps (default 1)
- `backend`: Array type (default `Array` for CPU)
- `src_type`: Source type — `:force` (default) or `:pressure`
"""
function simulate_elastic_wavefield(model::ElasticModel2D, geometry::Geometry,
                                    config::SimulationConfig;
                                    save_every::Int=1, backend=Array,
                                    src_type::Symbol=:force)
    grid = model.grid
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    nt = config.nt

    # FD coefficients
    coeffs = fd_coefficients(config.space_order)

    vp_max = maximum(model.vp)
    check_cfl(vp_max, dt, dx, dy; space_order=config.space_order)

    f0 = geometry.sources[1].wavelet.f0
    cpml = setup_cpml(grid, config, vp_max, f0; backend=backend)

    # Disable PML at the top boundary for free surface
    if config.free_surface
        _disable_cpml_left!(cpml.y, config.pml_points)
    end

    # Ghost cell padding for higher-order FD stencils
    pad = length(coeffs) - 1
    nxp, nyp = nx + 2 * pad, ny + 2 * pad

    if pad > 0
        vp_pad = pad_array(model.vp, pad)
        vs_pad = pad_array(model.vs, pad)
        rho_pad = pad_array(model.rho, pad)
        cpml = pad_cpml(cpml, pad)
    else
        vp_pad = model.vp
        vs_pad = model.vs
        rho_pad = model.rho
    end

    lambda = rho_pad .* (vp_pad .^ 2 .- 2.0 .* vs_pad .^ 2)
    mu = rho_pad .* vs_pad .^ 2
    state = ElasticState2D(nxp, nyp; backend=backend)

    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    rec_indices_pad = [(i + pad, j + pad) for (i, j) in rec_indices]
    seismograms_vx = zeros(Float64, nt, nrec)
    seismograms_vy = zeros(Float64, nt, nrec)

    n_snaps = length(1:save_every:nt)
    snaps_vx = zeros(Float64, nx, ny, n_snaps)
    snaps_vy = zeros(Float64, nx, ny, n_snaps)
    snap_idx = 0

    src = geometry.sources[1]
    isrc, jsrc = snap_to_grid(src.x, src.y, grid)
    isrc_pad, jsrc_pad = isrc + pad, jsrc + pad
    source_ts = compute_source_timeseries(src.wavelet, nt, dt)
    angle_rad = src.angle * π / 180.0

    for it in 1:nt
        elastic_stress_update!(state, lambda, mu, cpml, nxp, nyp, dx, dy, dt, coeffs)

        if config.free_surface
            elastic_apply_free_surface!(state, nxp, nyp, pad)
        end

        # Pressure source injection (after stress update, before velocity update)
        if src_type == :pressure
            elastic_apply_pressure_source!(state, source_ts[it], isrc_pad, jsrc_pad, dt)
        end

        elastic_velocity_update!(state, rho_pad, cpml, nxp, nyp, dx, dy, dt, coeffs)

        # Force source injection (after velocity update)
        if src_type == :force
            force_x = sin(angle_rad) * source_ts[it]
            force_y = cos(angle_rad) * source_ts[it]
            elastic_apply_source!(state, rho_pad, force_x, force_y, isrc_pad, jsrc_pad, dt)
        end
        elastic_apply_bc!(state, nxp, nyp; pad=pad, free_surface=config.free_surface)

        record_receivers!(seismograms_vx, state.vx, rec_indices_pad, it)
        record_receivers!(seismograms_vy, state.vy, rec_indices_pad, it)

        if mod(it - 1, save_every) == 0
            snap_idx += 1
            snaps_vx[:, :, snap_idx] .= @view state.vx[(pad+1):(pad+nx), (pad+1):(pad+ny)]
            snaps_vy[:, :, snap_idx] .= @view state.vy[(pad+1):(pad+nx), (pad+1):(pad+ny)]
        end
    end

    return seismograms_vx, seismograms_vy, snaps_vx, snaps_vy
end
