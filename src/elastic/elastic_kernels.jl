# Elastic kernel functions translated from seismic_CPML_2D_isotropic_second_order.f90

"""
    elastic_stress_update!(state, lambda, mu, cpml, nx, ny, dx, dy, dt, coeffs)

Update stress fields (sigma_xx, sigma_yy, sigma_xy) from velocity gradients.
Includes CPML memory variable updates.
"""
function elastic_stress_update!(state::ElasticState2D, lambda::AbstractMatrix,
                                mu::AbstractMatrix, cpml::CPML2D,
                                nx::Int, ny::Int, dx::Float64, dy::Float64, dt::Float64,
                                coeffs::NTuple{N,Float64}) where N
    hw = N

    # sigma_xx, sigma_yy update
    # dvx/dx at i+1/2: i = hw:(nx-hw)
    # dvy/dy backward at j (forward from j-1): j = (hw+1):(ny-hw+1)
    @inbounds for j in (hw+1):(ny-hw+1)
        for i in hw:(nx-hw)
            lambda_half_x = 0.5 * (lambda[i+1, j] + lambda[i, j])
            mu_half_x = 0.5 * (mu[i+1, j] + mu[i, j])
            lambda_plus_2mu = lambda_half_x + 2.0 * mu_half_x

            val_dvx_dx = fd_stencil_x(state.vx, i, j, coeffs, dx)
            val_dvy_dy = fd_stencil_y(state.vy, i, j-1, coeffs, dy)

            state.mem_dvx_dx[i, j] = cpml.x.b_half[i] * state.mem_dvx_dx[i, j] +
                                      cpml.x.a_half[i] * val_dvx_dx
            state.mem_dvy_dy[i, j] = cpml.y.b[j] * state.mem_dvy_dy[i, j] +
                                      cpml.y.a[j] * val_dvy_dy

            val_dvx_dx = val_dvx_dx / cpml.x.K_half[i] + state.mem_dvx_dx[i, j]
            val_dvy_dy = val_dvy_dy / cpml.y.K[j] + state.mem_dvy_dy[i, j]

            state.sigma_xx[i, j] += (lambda_plus_2mu * val_dvx_dx + lambda_half_x * val_dvy_dy) * dt
            state.sigma_yy[i, j] += (lambda_half_x * val_dvx_dx + lambda_plus_2mu * val_dvy_dy) * dt
        end
    end

    # sigma_xy update
    # dvy/dx backward at i (forward from i-1): i = (hw+1):(nx-hw+1)
    # dvx/dy at j+1/2: j = hw:(ny-hw)
    @inbounds for j in hw:(ny-hw)
        for i in (hw+1):(nx-hw+1)
            mu_half_y = 0.5 * (mu[i, j+1] + mu[i, j])

            val_dvy_dx = fd_stencil_x(state.vy, i-1, j, coeffs, dx)
            val_dvx_dy = fd_stencil_y(state.vx, i, j, coeffs, dy)

            state.mem_dvy_dx[i, j] = cpml.x.b[i] * state.mem_dvy_dx[i, j] +
                                      cpml.x.a[i] * val_dvy_dx
            state.mem_dvx_dy[i, j] = cpml.y.b_half[j] * state.mem_dvx_dy[i, j] +
                                      cpml.y.a_half[j] * val_dvx_dy

            val_dvy_dx = val_dvy_dx / cpml.x.K[i] + state.mem_dvy_dx[i, j]
            val_dvx_dy = val_dvx_dy / cpml.y.K_half[j] + state.mem_dvx_dy[i, j]

            state.sigma_xy[i, j] += mu_half_y * (val_dvy_dx + val_dvx_dy) * dt
        end
    end

    return nothing
end

"""
    elastic_velocity_update!(state, rho, cpml, nx, ny, dx, dy, dt, coeffs)

Update velocity fields (vx, vy) from stress divergence.
Includes CPML memory variable updates.
"""
function elastic_velocity_update!(state::ElasticState2D, rho::AbstractMatrix,
                                  cpml::CPML2D, nx::Int, ny::Int,
                                  dx::Float64, dy::Float64, dt::Float64,
                                  coeffs::NTuple{N,Float64}) where N
    hw = N

    # vx update
    # dsxx/dx backward at i (forward from i-1): i = (hw+1):(nx-hw+1)
    # dsxy/dy backward at j (forward from j-1): j = (hw+1):(ny-hw+1)
    @inbounds for j in (hw+1):(ny-hw+1)
        for i in (hw+1):(nx-hw+1)
            val_dsxx_dx = fd_stencil_x(state.sigma_xx, i-1, j, coeffs, dx)
            val_dsxy_dy = fd_stencil_y(state.sigma_xy, i, j-1, coeffs, dy)

            state.mem_dsxx_dx[i, j] = cpml.x.b[i] * state.mem_dsxx_dx[i, j] +
                                       cpml.x.a[i] * val_dsxx_dx
            state.mem_dsxy_dy[i, j] = cpml.y.b[j] * state.mem_dsxy_dy[i, j] +
                                       cpml.y.a[j] * val_dsxy_dy

            val_dsxx_dx = val_dsxx_dx / cpml.x.K[i] + state.mem_dsxx_dx[i, j]
            val_dsxy_dy = val_dsxy_dy / cpml.y.K[j] + state.mem_dsxy_dy[i, j]

            state.vx[i, j] += (val_dsxx_dx + val_dsxy_dy) * dt / rho[i, j]
        end
    end

    # vy update
    # dsxy/dx at i+1/2: i = hw:(nx-hw)
    # dsyy/dy at j+1/2: j = hw:(ny-hw)
    @inbounds for j in hw:(ny-hw)
        for i in hw:(nx-hw)
            rho_half = 0.25 * (rho[i, j] + rho[i+1, j] + rho[i+1, j+1] + rho[i, j+1])

            val_dsxy_dx = fd_stencil_x(state.sigma_xy, i, j, coeffs, dx)
            val_dsyy_dy = fd_stencil_y(state.sigma_yy, i, j, coeffs, dy)

            state.mem_dsxy_dx[i, j] = cpml.x.b_half[i] * state.mem_dsxy_dx[i, j] +
                                       cpml.x.a_half[i] * val_dsxy_dx
            state.mem_dsyy_dy[i, j] = cpml.y.b_half[j] * state.mem_dsyy_dy[i, j] +
                                       cpml.y.a_half[j] * val_dsyy_dy

            val_dsxy_dx = val_dsxy_dx / cpml.x.K_half[i] + state.mem_dsxy_dx[i, j]
            val_dsyy_dy = val_dsyy_dy / cpml.y.K_half[j] + state.mem_dsyy_dy[i, j]

            state.vy[i, j] += (val_dsxy_dx + val_dsyy_dy) * dt / rho_half
        end
    end

    return nothing
end

"""
    elastic_apply_source!(state, rho, force_x, force_y, isrc, jsrc, dt)

Inject force source at the given grid point.
"""
function elastic_apply_source!(state::ElasticState2D, rho::AbstractMatrix,
                               force_x::Float64, force_y::Float64,
                               isrc::Int, jsrc::Int, dt::Float64)
    i, j = isrc, jsrc
    nx, ny = size(rho)

    # vx source injection
    state.vx[i, j] += force_x * dt / rho[i, j]

    # vy source injection with interpolated density
    if i < nx && j < ny
        rho_half = 0.25 * (rho[i, j] + rho[i+1, j] + rho[i+1, j+1] + rho[i, j+1])
        state.vy[i, j] += force_y * dt / rho_half
    end

    return nothing
end

"""
    elastic_apply_pressure_source!(state, source_val, isrc, jsrc, dt)

Inject a pressure (explosive) source at the given grid point by adding
equally to the normal stress components σ_xx and σ_yy.
"""
function elastic_apply_pressure_source!(state::ElasticState2D,
                                         source_val::Float64,
                                         isrc::Int, jsrc::Int, dt::Float64)
    state.sigma_xx[isrc, jsrc] += source_val * dt
    state.sigma_yy[isrc, jsrc] += source_val * dt
    return nothing
end

"""
    elastic_apply_bc!(state, nx, ny; pad=0, free_surface=false)

Apply Dirichlet (rigid) boundary conditions at physical boundaries
(offset by `pad` ghost cells).  When `free_surface=true`, velocities
at the top boundary (lo_y) are left free.
"""
function elastic_apply_bc!(state::ElasticState2D, nx::Int, ny::Int;
                           pad::Int=0, free_surface::Bool=false)
    lo_x, hi_x = pad + 1, nx - pad
    lo_y, hi_y = pad + 1, ny - pad
    @inbounds for j in lo_y:hi_y
        state.vx[lo_x, j] = 0.0
        state.vx[hi_x, j] = 0.0
        state.vy[lo_x, j] = 0.0
        state.vy[hi_x, j] = 0.0
    end
    @inbounds for i in lo_x:hi_x
        if !free_surface
            state.vx[i, lo_y] = 0.0
            state.vy[i, lo_y] = 0.0
        end
        state.vx[i, hi_y] = 0.0
        state.vy[i, hi_y] = 0.0
    end
    return nothing
end

"""
    elastic_apply_free_surface!(state, nx, ny, pad)

Apply stress-free boundary condition at the top surface (j = pad+1).
Sets σ_yy = 0 at the surface and mirrors stresses anti-symmetrically
into ghost cells above using the DENISE method-of-imaging formulas:
  σ_yy[i, j_surf - m] = -σ_yy[i, j_surf + m]
  σ_xy[i, j_surf - m] = -σ_xy[i, j_surf + m - 1]   (half-grid offset)
"""
function elastic_apply_free_surface!(state::ElasticState2D, nx::Int, ny::Int, pad::Int)
    j_surf = pad + 1
    # Zero normal stress at the surface
    @inbounds for i in 1:nx
        state.sigma_yy[i, j_surf] = 0.0
    end
    # Mirror stresses into ghost cells
    @inbounds for m in 1:pad
        for i in 1:nx
            state.sigma_yy[i, j_surf - m] = -state.sigma_yy[i, j_surf + m]
            state.sigma_xy[i, j_surf - m] = -state.sigma_xy[i, j_surf + m - 1]
        end
    end
    return nothing
end
