"""
    snap_to_grid(x, y, grid::Grid2D) -> (i, j)

Convert physical coordinates (meters) to nearest grid indices (1-based).
"""
function snap_to_grid(x::Real, y::Real, grid::Grid2D)
    i = round(Int, x / grid.dx) + 1
    j = round(Int, y / grid.dy) + 1
    i = clamp(i, 1, grid.nx)
    j = clamp(j, 1, grid.ny)
    return (i, j)
end

"""
    grid_coords(i, j, grid::Grid2D) -> (x, y)

Convert grid indices to physical coordinates (meters).
"""
function grid_coords(i::Int, j::Int, grid::Grid2D)
    x = (i - 1) * grid.dx
    y = (j - 1) * grid.dy
    return (x, y)
end
