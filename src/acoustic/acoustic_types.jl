struct AcousticModel2D{T, A<:AbstractMatrix{T}}
    vp::A       # P-wave velocity (nx, ny)
    rho::A      # density (nx, ny)
    grid::Grid2D
end

function AcousticModel2D(vp::AbstractMatrix, rho::AbstractMatrix, grid::Grid2D)
    @assert size(vp) == (grid.nx, grid.ny) "vp size must match grid dimensions"
    @assert size(rho) == (grid.nx, grid.ny) "rho size must match grid dimensions"
    AcousticModel2D{eltype(vp), typeof(vp)}(vp, rho, grid)
end

mutable struct AcousticState2D{A<:AbstractMatrix}
    pressure_past::A
    pressure_present::A
    pressure_future::A
    pressure_xx::A
    pressure_yy::A
    dpressurexx_dx::A
    dpressureyy_dy::A
    # PML memory variables
    mem_dp_dx::A
    mem_dp_dy::A
    mem_dpxx_dx::A
    mem_dpyy_dy::A
end

function AcousticState2D(nx::Int, ny::Int; backend=Array)
    z() = backend(zeros(Float64, nx, ny))
    AcousticState2D(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end
