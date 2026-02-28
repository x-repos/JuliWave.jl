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

    # Setup CPML coefficients
    cpml = setup_cpml(grid, config, vp_max, f0; backend=backend)

    # Precompute kappa (bulk modulus) = rho * vp²
    kappa = model.rho .* model.vp .^ 2

    # Initialize state
    state = AcousticState2D(nx, ny; backend=backend)

    # Snap receivers to grid
    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
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

        # Source grid position
        isrc, jsrc = snap_to_grid(src.x, src.y, grid)
        cp_src = model.vp[isrc, jsrc]

        # Precompute source time series
        source_ts = compute_source_timeseries(src.wavelet, nt, dt)

        # Time stepping
        for it in 1:nt
            # First spatial derivatives / rho
            acoustic_first_derivatives!(state, model, cpml, nx, ny, dx, dy, coeffs)

            # Second spatial derivatives
            acoustic_second_derivatives!(state, cpml, nx, ny, dx, dy, coeffs)

            # Time update with source injection and Dirichlet BCs
            acoustic_time_update!(state, kappa, source_ts[it], isrc, jsrc, dt, nx, ny, cp_src)

            # Record receivers
            record_receivers!(seismograms, state.pressure_future, rec_indices, it)

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
    kappa = model.rho .* model.vp .^ 2
    state = AcousticState2D(nx, ny; backend=backend)

    rec_indices = snap_receivers(geometry.receivers, grid)
    nrec = length(rec_indices)
    seismograms = zeros(Float64, nt, nrec)

    n_snaps = length(1:save_every:nt)
    snapshots = zeros(Float64, nx, ny, n_snaps)
    snap_idx = 0

    src = geometry.sources[1]
    isrc, jsrc = snap_to_grid(src.x, src.y, grid)
    cp_src = model.vp[isrc, jsrc]
    source_ts = compute_source_timeseries(src.wavelet, nt, dt)

    for it in 1:nt
        acoustic_first_derivatives!(state, model, cpml, nx, ny, dx, dy, coeffs)
        acoustic_second_derivatives!(state, cpml, nx, ny, dx, dy, coeffs)
        acoustic_time_update!(state, kappa, source_ts[it], isrc, jsrc, dt, nx, ny, cp_src)
        record_receivers!(seismograms, state.pressure_future, rec_indices, it)

        if mod(it - 1, save_every) == 0
            snap_idx += 1
            snapshots[:, :, snap_idx] .= state.pressure_future
        end

        state.pressure_past, state.pressure_present, state.pressure_future =
            state.pressure_present, state.pressure_future, state.pressure_past
    end

    return seismograms, snapshots
end
