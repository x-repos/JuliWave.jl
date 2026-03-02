"""
    simulate_acoustic(model, geometry, config; backend=Array) -> Matrix{Float64}

Run a full 2D acoustic forward simulation with C-PML absorbing boundaries.
Returns seismograms as a matrix of size (nt, nrec).

Translated from seismic_CPML_2D_pressure_second_order.f90
"""
function simulate_acoustic(model::AcousticModel2D, geometry::Geometry,
                           config::SimulationConfig; backend=Array)
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

    # Get dominant frequency from first source wavelet
    f0 = geometry.sources[1].wavelet.f0

    # Setup CPML coefficients on original grid
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
        rho_pad = pad_array(model.rho, pad)
        grid_pad = Grid2D(nxp, nyp, dx, dy)
        model_pad = AcousticModel2D(vp_pad, rho_pad, grid_pad)
        cpml = pad_cpml(cpml, pad)
    else
        model_pad = model
        rho_pad = model.rho
        vp_pad = model.vp
    end

    # Precompute kappa (bulk modulus) = rho * vp²
    kappa = rho_pad .* vp_pad .^ 2

    # Initialize state with padded dimensions
    state = AcousticState2D(nxp, nyp; backend=backend)

    # Snap receivers to grid (physical coordinates)
    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    rec_indices_pad = [(i + pad, j + pad) for (i, j) in rec_indices]
    seismograms = zeros(Float64, nt, nrec)

    # Process each source (sequential shot gather)
    for src in geometry.sources
        # Reset state
        state.pressure_past .= 0.0
        state.pressure_present .= 0.0
        state.pressure_future .= 0.0
        state.pressure_xx .= 0.0
        state.pressure_yy .= 0.0
        state.dpressurexx_dx .= 0.0
        state.dpressureyy_dy .= 0.0
        state.mem_dp_dx .= 0.0
        state.mem_dp_dy .= 0.0
        state.mem_dpxx_dx .= 0.0
        state.mem_dpyy_dy .= 0.0

        # Source grid position (physical grid, then offset)
        isrc, jsrc = snap_to_grid(src.x, src.y, grid)
        cp_src = model.vp[isrc, jsrc]
        isrc_pad, jsrc_pad = isrc + pad, jsrc + pad

        # Precompute source time series
        source_ts = compute_source_timeseries(src.wavelet, nt, dt)

        # Time stepping
        for it in 1:nt
            # First spatial derivatives / rho
            acoustic_first_derivatives!(state, model_pad, cpml, nxp, nyp, dx, dy, coeffs)

            # Second spatial derivatives
            acoustic_second_derivatives!(state, cpml, nxp, nyp, dx, dy, coeffs)

            # Time update with source injection and Dirichlet BCs
            acoustic_time_update!(state, kappa, source_ts[it], isrc_pad, jsrc_pad,
                                  dt, nxp, nyp, cp_src; pad=pad)

            # Free surface: p=0 at top + mirror into ghost cells
            if config.free_surface
                acoustic_apply_free_surface!(state, nxp, nyp, pad)
            end

            # Record receivers
            record_receivers!(seismograms, state.pressure_future, rec_indices_pad, it)

            # Rotate time levels
            state.pressure_past, state.pressure_present, state.pressure_future =
                state.pressure_present, state.pressure_future, state.pressure_past
        end
    end

    return seismograms
end

"""
    simulate_acoustic_wavefield(model, geometry, config; save_every=1, backend=Array)

Run acoustic simulation and return both seismograms and wavefield snapshots.
Returns (seismograms, snapshots) where snapshots is a 3D array (nx, ny, n_snaps).
"""
function simulate_acoustic_wavefield(model::AcousticModel2D, geometry::Geometry,
                                     config::SimulationConfig;
                                     save_every::Int=1, backend=Array)
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
        rho_pad = pad_array(model.rho, pad)
        grid_pad = Grid2D(nxp, nyp, dx, dy)
        model_pad = AcousticModel2D(vp_pad, rho_pad, grid_pad)
        cpml = pad_cpml(cpml, pad)
    else
        model_pad = model
        rho_pad = model.rho
        vp_pad = model.vp
    end

    kappa = rho_pad .* vp_pad .^ 2
    state = AcousticState2D(nxp, nyp; backend=backend)

    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    rec_indices_pad = [(i + pad, j + pad) for (i, j) in rec_indices]
    seismograms = zeros(Float64, nt, nrec)

    n_snaps = length(1:save_every:nt)
    snapshots = zeros(Float64, nx, ny, n_snaps)
    snap_idx = 0

    src = geometry.sources[1]
    isrc, jsrc = snap_to_grid(src.x, src.y, grid)
    cp_src = model.vp[isrc, jsrc]
    isrc_pad, jsrc_pad = isrc + pad, jsrc + pad
    source_ts = compute_source_timeseries(src.wavelet, nt, dt)

    for it in 1:nt
        acoustic_first_derivatives!(state, model_pad, cpml, nxp, nyp, dx, dy, coeffs)
        acoustic_second_derivatives!(state, cpml, nxp, nyp, dx, dy, coeffs)
        acoustic_time_update!(state, kappa, source_ts[it], isrc_pad, jsrc_pad,
                              dt, nxp, nyp, cp_src; pad=pad)

        if config.free_surface
            acoustic_apply_free_surface!(state, nxp, nyp, pad)
        end

        record_receivers!(seismograms, state.pressure_future, rec_indices_pad, it)

        if mod(it - 1, save_every) == 0
            snap_idx += 1
            snapshots[:, :, snap_idx] .= @view state.pressure_future[(pad+1):(pad+nx), (pad+1):(pad+ny)]
        end

        state.pressure_past, state.pressure_present, state.pressure_future =
            state.pressure_present, state.pressure_future, state.pressure_past
    end

    return seismograms, snapshots
end
