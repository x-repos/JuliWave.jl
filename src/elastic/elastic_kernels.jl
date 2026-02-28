# Elastic kernel functions translated from seismic_CPML_2D_isotropic_second_order.f90

"""
    elastic_stress_update!(state, lambda, mu, cpml, nx, ny, dx, dy, dt)

Update stress fields (sigma_xx, sigma_yy, sigma_xy) from velocity gradients.
Includes CPML memory variable updates.
"""
function elastic_stress_update!(state::ElasticState2D, lambda::AbstractMatrix,
                                mu::AbstractMatrix, cpml::CPML2D,
                                nx::Int, ny::Int, dx::Float64, dy::Float64, dt::Float64)
    # sigma_xx, sigma_yy update (i=1:NX-1, j=2:NY)
    @inbounds for j in 2:ny
        for i in 1:(nx-1)
            # Interpolate material parameters at staggered location
            lambda_half_x = 0.5 * (lambda[i+1, j] + lambda[i, j])
            mu_half_x = 0.5 * (mu[i+1, j] + mu[i, j])
            lambda_plus_2mu = lambda_half_x + 2.0 * mu_half_x

            val_dvx_dx = (state.vx[i+1, j] - state.vx[i, j]) / dx
            val_dvy_dy = (state.vy[i, j] - state.vy[i, j-1]) / dy

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

    # sigma_xy update (i=2:NX, j=1:NY-1)
    @inbounds for j in 1:(ny-1)
        for i in 2:nx
            mu_half_y = 0.5 * (mu[i, j+1] + mu[i, j])

            val_dvy_dx = (state.vy[i, j] - state.vy[i-1, j]) / dx
            val_dvx_dy = (state.vx[i, j+1] - state.vx[i, j]) / dy

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
    elastic_velocity_update!(state, rho, cpml, nx, ny, dx, dy, dt)

Update velocity fields (vx, vy) from stress divergence.
Includes CPML memory variable updates.
"""
function elastic_velocity_update!(state::ElasticState2D, rho::AbstractMatrix,
                                  cpml::CPML2D, nx::Int, ny::Int,
                                  dx::Float64, dy::Float64, dt::Float64)
    # vx update (i=2:NX, j=2:NY)
    @inbounds for j in 2:ny
        for i in 2:nx
            val_dsxx_dx = (state.sigma_xx[i, j] - state.sigma_xx[i-1, j]) / dx
            val_dsxy_dy = (state.sigma_xy[i, j] - state.sigma_xy[i, j-1]) / dy

            state.mem_dsxx_dx[i, j] = cpml.x.b[i] * state.mem_dsxx_dx[i, j] +
                                       cpml.x.a[i] * val_dsxx_dx
            state.mem_dsxy_dy[i, j] = cpml.y.b[j] * state.mem_dsxy_dy[i, j] +
                                       cpml.y.a[j] * val_dsxy_dy

            val_dsxx_dx = val_dsxx_dx / cpml.x.K[i] + state.mem_dsxx_dx[i, j]
            val_dsxy_dy = val_dsxy_dy / cpml.y.K[j] + state.mem_dsxy_dy[i, j]

            state.vx[i, j] += (val_dsxx_dx + val_dsxy_dy) * dt / rho[i, j]
        end
    end

    # vy update (i=1:NX-1, j=1:NY-1)
    @inbounds for j in 1:(ny-1)
        for i in 1:(nx-1)
            # Interpolate density at staggered location
            rho_half = 0.25 * (rho[i, j] + rho[i+1, j] + rho[i+1, j+1] + rho[i, j+1])

            val_dsxy_dx = (state.sigma_xy[i+1, j] - state.sigma_xy[i, j]) / dx
            val_dsyy_dy = (state.sigma_yy[i, j+1] - state.sigma_yy[i, j]) / dy

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
    elastic_apply_bc!(state, nx, ny)

Apply Dirichlet (rigid) boundary conditions on all edges.
"""
function elastic_apply_bc!(state::ElasticState2D, nx::Int, ny::Int)
    @inbounds for j in 1:ny
        state.vx[1, j] = 0.0
        state.vx[nx, j] = 0.0
        state.vy[1, j] = 0.0
        state.vy[nx, j] = 0.0
    end
    @inbounds for i in 1:nx
        state.vx[i, 1] = 0.0
        state.vx[i, ny] = 0.0
        state.vy[i, 1] = 0.0
        state.vy[i, ny] = 0.0
    end
    return nothing
end
