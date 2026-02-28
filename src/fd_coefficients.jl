# Staggered-grid finite-difference coefficients for first derivatives at half-points.
#
# For order 2N, the derivative at i+1/2 uses N points on each side:
#   df/dx|_{i+1/2} ≈ (1/dx) * Σ_{k=1}^{N} c_k * (f[i+k] - f[i+1-k])
#
# Coefficients derived from the Fornberg staggered-grid formula:
#   d_k = Π_{j≠k} (-y_j) / (y_k - y_j)   where y_k = (2k-1)²
#   c_k = d_k / (2k-1)

"""
    _compute_staggered_coeffs(N::Int) -> NTuple{N, Float64}

Compute staggered-grid FD coefficients for order 2N using Fornberg's formula.
"""
function _compute_staggered_coeffs(N::Int)
    y = ntuple(k -> (2k - 1)^2, N)
    coeffs = ntuple(N) do k
        dk = 1.0
        for j in 1:N
            j == k && continue
            dk *= (-y[j]) / (y[k] - y[j])
        end
        dk / (2k - 1)
    end
    return coeffs
end

# Precompute all supported coefficient tuples as constants
const _FD_COEFFS_2  = _compute_staggered_coeffs(1)
const _FD_COEFFS_4  = _compute_staggered_coeffs(2)
const _FD_COEFFS_6  = _compute_staggered_coeffs(3)
const _FD_COEFFS_8  = _compute_staggered_coeffs(4)
const _FD_COEFFS_10 = _compute_staggered_coeffs(5)
const _FD_COEFFS_12 = _compute_staggered_coeffs(6)
const _FD_COEFFS_14 = _compute_staggered_coeffs(7)
const _FD_COEFFS_16 = _compute_staggered_coeffs(8)

# Val-dispatched accessors for zero-cost abstraction (no runtime branching)
@inline fd_coefficients(::Val{2})  = _FD_COEFFS_2
@inline fd_coefficients(::Val{4})  = _FD_COEFFS_4
@inline fd_coefficients(::Val{6})  = _FD_COEFFS_6
@inline fd_coefficients(::Val{8})  = _FD_COEFFS_8
@inline fd_coefficients(::Val{10}) = _FD_COEFFS_10
@inline fd_coefficients(::Val{12}) = _FD_COEFFS_12
@inline fd_coefficients(::Val{14}) = _FD_COEFFS_14
@inline fd_coefficients(::Val{16}) = _FD_COEFFS_16

"""
    fd_coefficients(order::Int) -> NTuple{N, Float64}

Return staggered-grid FD coefficients for the given spatial order (2, 4, 6, 8, 10, 12, 14, 16).
Order 2N uses N coefficients.
"""
function fd_coefficients(order::Int)
    order == 2  && return _FD_COEFFS_2
    order == 4  && return _FD_COEFFS_4
    order == 6  && return _FD_COEFFS_6
    order == 8  && return _FD_COEFFS_8
    order == 10 && return _FD_COEFFS_10
    order == 12 && return _FD_COEFFS_12
    order == 14 && return _FD_COEFFS_14
    order == 16 && return _FD_COEFFS_16
    error("Unsupported FD order $order. Supported orders: 2, 4, 6, 8, 10, 12, 14, 16")
end

"""
    fd_stencil_x(f, i, j, coeffs::NTuple{N,Float64}, dx) -> Float64

Compute df/dx at position (i+1/2, j) using staggered-grid FD stencil along x.
For backward derivative at (i-1/2, j), call with i-1.
"""
@inline function fd_stencil_x(f, i, j, coeffs::NTuple{N,Float64}, dx::Float64) where N
    val = zero(Float64)
    for k in 1:N
        @inbounds val += coeffs[k] * (f[i+k, j] - f[i+1-k, j])
    end
    return val / dx
end

"""
    fd_stencil_y(f, i, j, coeffs::NTuple{N,Float64}, dy) -> Float64

Compute df/dy at position (i, j+1/2) using staggered-grid FD stencil along y.
For backward derivative at (i, j-1/2), call with j-1.
"""
@inline function fd_stencil_y(f, i, j, coeffs::NTuple{N,Float64}, dy::Float64) where N
    val = zero(Float64)
    for k in 1:N
        @inbounds val += coeffs[k] * (f[i, j+k] - f[i, j+1-k])
    end
    return val / dy
end
