struct ElasticModel2D{T, A<:AbstractMatrix{T}}
    vp::A       # P-wave velocity (nx, ny)
    vs::A       # S-wave velocity (nx, ny)
    rho::A      # density (nx, ny)
    grid::Grid2D
end

function ElasticModel2D(vp::AbstractMatrix, vs::AbstractMatrix, rho::AbstractMatrix, grid::Grid2D)
    @assert size(vp) == (grid.nx, grid.ny) "vp size must match grid dimensions"
    @assert size(vs) == (grid.nx, grid.ny) "vs size must match grid dimensions"
    @assert size(rho) == (grid.nx, grid.ny) "rho size must match grid dimensions"
    ElasticModel2D{eltype(vp), typeof(vp)}(vp, vs, rho, grid)
end

mutable struct ElasticState2D{A<:AbstractMatrix}
    vx::A
    vy::A
    sigma_xx::A
    sigma_yy::A
    sigma_xy::A
    # PML memory variables (8 arrays)
    mem_dvx_dx::A
    mem_dvx_dy::A
    mem_dvy_dx::A
    mem_dvy_dy::A
    mem_dsxx_dx::A
    mem_dsyy_dy::A
    mem_dsxy_dx::A
    mem_dsxy_dy::A
end

function ElasticState2D(nx::Int, ny::Int; backend=Array)
    z() = backend(zeros(Float64, nx, ny))
    ElasticState2D(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end
