"""
    snap_receivers(receivers, grid) -> Vector{Tuple{Int,Int}}

Snap all receivers to nearest grid points.
"""
function snap_receivers(receivers::Vector{Receiver}, grid::Grid2D)
    return [snap_to_grid(r.x, r.y, grid) for r in receivers]
end

"""
    record_receivers!(seismograms, field, rec_indices, it)

Extract field values at receiver locations for time step `it`.
"""
function record_receivers!(seismograms::AbstractMatrix, field::AbstractMatrix,
                           rec_indices::Vector{Tuple{Int,Int}}, it::Int)
    for (irec, (i, j)) in enumerate(rec_indices)
        seismograms[it, irec] = field[i, j]
    end
    return nothing
end
