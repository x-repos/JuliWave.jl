# Acoustic kernel functions for GPU-compatible stencil operations.
# These use explicit loops for Enzyme compatibility.

"""
    acoustic_first_derivatives!(state, model, cpml, nx, ny, dx, dy, coeffs)

Compute first spatial derivatives of pressure divided by density, with CPML.
Translated from Fortran: dp/dx at half-grid, dp/dy at half-grid.
"""
function acoustic_first_derivatives!(state::AcousticState2D, model::AcousticModel2D,
                                     cpml::CPML2D, nx::Int, ny::Int, dx::Float64, dy::Float64,
                                     coeffs::NTuple{N,Float64}) where N
    p = state.pressure_present
    rho = model.rho
    hw = N  # half-width of stencil

    # dp/dx at i+1/2 (i=hw:nx-hw, j=1:ny)
    @inbounds for j in 1:ny
        for i in hw:(nx-hw)
            val = fd_stencil_x(p, i, j, coeffs, dx)

            state.mem_dp_dx[i, j] = cpml.x.b_half[i] * state.mem_dp_dx[i, j] +
                                     cpml.x.a_half[i] * val

            val = val / cpml.x.K_half[i] + state.mem_dp_dx[i, j]

            rho_half_x = 0.5 * (rho[i+1, j] + rho[i, j])
            state.pressure_xx[i, j] = val / rho_half_x
        end
    end

    # dp/dy at j+1/2 (i=1:nx, j=hw:ny-hw)
    @inbounds for j in hw:(ny-hw)
        for i in 1:nx
            val = fd_stencil_y(p, i, j, coeffs, dy)

            state.mem_dp_dy[i, j] = cpml.y.b_half[j] * state.mem_dp_dy[i, j] +
                                     cpml.y.a_half[j] * val

            val = val / cpml.y.K_half[j] + state.mem_dp_dy[i, j]

            rho_half_y = 0.5 * (rho[i, j+1] + rho[i, j])
            state.pressure_yy[i, j] = val / rho_half_y
        end
    end

    return nothing
end

"""
    acoustic_second_derivatives!(state, cpml, nx, ny, dx, dy, coeffs)

Compute second spatial derivatives with CPML.
d(pressure_xx)/dx and d(pressure_yy)/dy at full grid points.
"""
function acoustic_second_derivatives!(state::AcousticState2D, cpml::CPML2D,
                                      nx::Int, ny::Int, dx::Float64, dy::Float64,
                                      coeffs::NTuple{N,Float64}) where N
    hw = N

    # d(pressure_xx)/dx at i (backward = forward from i-1)
    # i = (hw+1):(nx-hw+1)
    @inbounds for j in 1:ny
        for i in (hw+1):(nx-hw+1)
            val = fd_stencil_x(state.pressure_xx, i-1, j, coeffs, dx)

            state.mem_dpxx_dx[i, j] = cpml.x.b[i] * state.mem_dpxx_dx[i, j] +
                                       cpml.x.a[i] * val

            val = val / cpml.x.K[i] + state.mem_dpxx_dx[i, j]

            state.dpressurexx_dx[i, j] = val
        end
    end

    # d(pressure_yy)/dy at j (backward = forward from j-1)
    # j = (hw+1):(ny-hw+1)
    @inbounds for j in (hw+1):(ny-hw+1)
        for i in 1:nx
            val = fd_stencil_y(state.pressure_yy, i, j-1, coeffs, dy)

            state.mem_dpyy_dy[i, j] = cpml.y.b[j] * state.mem_dpyy_dy[i, j] +
                                       cpml.y.a[j] * val

            val = val / cpml.y.K[j] + state.mem_dpyy_dy[i, j]

            state.dpressureyy_dy[i, j] = val
        end
    end

    return nothing
end

"""
    acoustic_time_update!(state, kappa, source_val, isrc, jsrc, dt, nx, ny, cp_src)

Apply the time evolution scheme:
    p_future = -p_past + 2*p_present + dt² * (d²p * kappa + 4π * cp² * source * δ)
Then apply Dirichlet BCs.
"""
function acoustic_time_update!(state::AcousticState2D, kappa::AbstractMatrix,
                               source_val::Float64, isrc::Int, jsrc::Int,
                               dt::Float64, nx::Int, ny::Int, cp_src::Float64)
    dt2 = dt * dt
    src_factor = 4.0 * π * cp_src^2 * source_val

    @inbounds for j in 1:ny
        for i in 1:nx
            laplacian = state.dpressurexx_dx[i, j] + state.dpressureyy_dy[i, j]
            src_contrib = (i == isrc && j == jsrc) ? src_factor : 0.0
            state.pressure_future[i, j] = -state.pressure_past[i, j] +
                                            2.0 * state.pressure_present[i, j] +
                                            dt2 * (laplacian * kappa[i, j] + src_contrib)
        end
    end

    # Dirichlet boundary conditions
    @inbounds for j in 1:ny
        state.pressure_future[1, j] = 0.0
        state.pressure_future[nx, j] = 0.0
    end
    @inbounds for i in 1:nx
        state.pressure_future[i, 1] = 0.0
        state.pressure_future[i, ny] = 0.0
    end

    return nothing
end
