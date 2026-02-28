"""
    check_cfl(vmax, dt, dx, dy; space_order=2)

Check the Courant-Friedrichs-Lewy stability condition.
For staggered-grid order 2N in 2D:
    C = vmax * dt * sqrt(ndim) * sum(|c_k|) * sqrt(1/dx² + 1/dy²) < 1

The stability factor accounts for the sum of absolute FD coefficients.
"""
function check_cfl(vmax::Real, dt::Real, dx::Real, dy::Real; space_order::Int=2)
    coeffs = fd_coefficients(space_order)
    coeff_sum = sum(abs, coeffs)
    courant = vmax * dt * coeff_sum * sqrt(1.0 / dx^2 + 1.0 / dy^2)
    if courant > 1.0
        error("CFL condition violated: Courant number = $courant > 1.0. " *
              "Reduce dt or increase grid spacing.")
    end
    return courant
end

"""
    suggest_dt(vmax, dx, dy; courant_target=0.5, space_order=2)

Suggest a stable time step for given grid spacing and max velocity.
Accounts for higher-order FD stencil stability requirements.
"""
function suggest_dt(vmax::Real, dx::Real, dy::Real; courant_target=0.5, space_order::Int=2)
    coeffs = fd_coefficients(space_order)
    coeff_sum = sum(abs, coeffs)
    return courant_target / (vmax * coeff_sum * sqrt(1.0 / dx^2 + 1.0 / dy^2))
end

"""
    select_backend(use_gpu::Bool)

Return appropriate array constructor based on GPU availability.
"""
function select_backend(use_gpu::Bool)
    if use_gpu
        return _get_cuda_array()
    else
        return Array
    end
end

function _get_cuda_array()
    try
        @eval using CUDA
        if CUDA.functional()
            return CUDA.CuArray
        else
            @warn "CUDA not functional, falling back to CPU"
            return Array
        end
    catch
        @warn "CUDA.jl not available, falling back to CPU"
        return Array
    end
end
