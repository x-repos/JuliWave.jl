struct CPMLCoefficients1D{V<:AbstractVector}
    a::V
    b::V
    K::V
    a_half::V
    b_half::V
    K_half::V
end

struct CPML2D{V}
    x::CPMLCoefficients1D{V}
    y::CPMLCoefficients1D{V}
end

"""
    setup_cpml(grid, config, max_velocity; backend=Array) -> CPML2D

Compute C-PML damping profiles for both X and Y directions.
Translated from the Fortran reference (Komatitsch CPML codes).

The algorithm:
- d0 = -(N+1) * Vmax * ln(Rcoef) / (2 * thickness)
- d(x) = d0 * (x_normalized)^NPOWER
- K(x) = 1 + (K_MAX - 1) * (x_normalized)^NPOWER
- alpha(x) = ALPHA_MAX * (1 - x_normalized)
- b(x) = exp(-(d/K + alpha) * dt)
- a(x) = d * (b - 1) / (K * (d + K * alpha))   [when |d| > 1e-6]
"""
function setup_cpml(grid::Grid2D, config::SimulationConfig, max_velocity::Float64;
                    backend=Array)
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    npoints = config.pml_points
    Rcoef = config.pml_Rcoef
    npower = config.pml_npower
    K_MAX = config.pml_kmax
    f0_approx = max_velocity / (10.0 * min(dx, dy))  # estimate dominant freq for alpha
    ALPHA_MAX = 2.0 * π * (f0_approx / 2.0)

    x_coeffs = _compute_cpml_1d(nx, dx, npoints, npower, K_MAX, ALPHA_MAX, Rcoef, max_velocity, dt; backend=backend)
    y_coeffs = _compute_cpml_1d(ny, dy, npoints, npower, K_MAX, ALPHA_MAX, Rcoef, max_velocity, dt; backend=backend)

    return CPML2D(x_coeffs, y_coeffs)
end

"""
    setup_cpml(grid, config, max_velocity, f0; backend=Array) -> CPML2D

Version with explicit dominant frequency for ALPHA_MAX computation.
"""
function setup_cpml(grid::Grid2D, config::SimulationConfig, max_velocity::Float64,
                    f0::Float64; backend=Array)
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = config.dt
    npoints = config.pml_points
    Rcoef = config.pml_Rcoef
    npower = config.pml_npower
    K_MAX = config.pml_kmax
    ALPHA_MAX = 2.0 * π * (f0 / 2.0)

    x_coeffs = _compute_cpml_1d(nx, dx, npoints, npower, K_MAX, ALPHA_MAX, Rcoef, max_velocity, dt; backend=backend)
    y_coeffs = _compute_cpml_1d(ny, dy, npoints, npower, K_MAX, ALPHA_MAX, Rcoef, max_velocity, dt; backend=backend)

    return CPML2D(x_coeffs, y_coeffs)
end

function _compute_cpml_1d(n::Int, delta::Float64, npoints::Int, npower::Float64,
                          K_MAX::Float64, ALPHA_MAX::Float64, Rcoef::Float64,
                          vmax::Float64, dt::Float64; backend=Array)
    thickness = npoints * delta
    d0 = -(npower + 1) * vmax * log(Rcoef) / (2.0 * thickness)

    origin_left = thickness
    origin_right = (n - 1) * delta - thickness

    d_arr = zeros(Float64, n)
    d_half = zeros(Float64, n)
    K_arr = ones(Float64, n)
    K_half_arr = ones(Float64, n)
    alpha_arr = zeros(Float64, n)
    alpha_half = zeros(Float64, n)
    a_arr = zeros(Float64, n)
    a_half_arr = zeros(Float64, n)
    b_arr = zeros(Float64, n)
    b_half_arr = zeros(Float64, n)

    for i in 1:n
        xval = delta * (i - 1)

        # Left edge - grid points
        abscissa = origin_left - xval
        if abscissa >= 0.0
            norm = abscissa / thickness
            d_arr[i] = d0 * norm^npower
            K_arr[i] = 1.0 + (K_MAX - 1.0) * norm^npower
            alpha_arr[i] = ALPHA_MAX * (1.0 - norm)
        end

        # Left edge - half grid points
        abscissa = origin_left - (xval + delta / 2.0)
        if abscissa >= 0.0
            norm = abscissa / thickness
            d_half[i] = d0 * norm^npower
            K_half_arr[i] = 1.0 + (K_MAX - 1.0) * norm^npower
            alpha_half[i] = ALPHA_MAX * (1.0 - norm)
        end

        # Right edge - grid points
        abscissa = xval - origin_right
        if abscissa >= 0.0
            norm = abscissa / thickness
            d_arr[i] = d0 * norm^npower
            K_arr[i] = 1.0 + (K_MAX - 1.0) * norm^npower
            alpha_arr[i] = ALPHA_MAX * (1.0 - norm)
        end

        # Right edge - half grid points
        abscissa = xval + delta / 2.0 - origin_right
        if abscissa >= 0.0
            norm = abscissa / thickness
            d_half[i] = d0 * norm^npower
            K_half_arr[i] = 1.0 + (K_MAX - 1.0) * norm^npower
            alpha_half[i] = ALPHA_MAX * (1.0 - norm)
        end

        # Clamp alpha to non-negative
        alpha_arr[i] = max(alpha_arr[i], 0.0)
        alpha_half[i] = max(alpha_half[i], 0.0)

        # Compute b and a coefficients
        b_arr[i] = exp(-(d_arr[i] / K_arr[i] + alpha_arr[i]) * dt)
        b_half_arr[i] = exp(-(d_half[i] / K_half_arr[i] + alpha_half[i]) * dt)

        if abs(d_arr[i]) > 1.0e-6
            a_arr[i] = d_arr[i] * (b_arr[i] - 1.0) /
                        (K_arr[i] * (d_arr[i] + K_arr[i] * alpha_arr[i]))
        end
        if abs(d_half[i]) > 1.0e-6
            a_half_arr[i] = d_half[i] * (b_half_arr[i] - 1.0) /
                             (K_half_arr[i] * (d_half[i] + K_half_arr[i] * alpha_half[i]))
        end
    end

    return CPMLCoefficients1D(
        backend(a_arr), backend(b_arr), backend(K_arr),
        backend(a_half_arr), backend(b_half_arr), backend(K_half_arr)
    )
end
