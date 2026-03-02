# NPZ I/O utilities for saving/loading simulation data

using NPZ

"""
    save_npz(filename, data::Dict)

Save a dictionary of arrays to an NPZ file.
"""
function save_npz(filename::AbstractString, data::Dict)
    npzwrite(filename, data)
end

"""
    load_npz(filename) -> Dict{String, Array}

Load an NPZ file and return a dictionary of arrays.
"""
function load_npz(filename::AbstractString)
    return npzread(filename)
end

"""
    save_seismograms(filename, seismograms...; names, dt, kwargs...)

Save seismogram arrays to an NPZ file with metadata.

# Arguments
- `filename`: Output NPZ file path
- `seismograms...`: One or more seismogram arrays (nt × nrec)
- `names`: Vector of names for each seismogram (e.g., ["vx", "vy"])
- `dt`: Time step size
- `kwargs...`: Additional metadata arrays to include
"""
function save_seismograms(filename::AbstractString, seismograms...;
                          names::Vector{String}, dt::Float64, kwargs...)
    data = Dict{String, Any}()
    for (name, seis) in zip(names, seismograms)
        data[name] = seis
    end
    data["dt"] = [dt]
    for (k, v) in kwargs
        data[string(k)] = v isa AbstractArray ? v : [v]
    end
    npzwrite(filename, data)
end

"""
    load_seismograms(filename) -> Dict{String, Any}

Load seismograms from an NPZ file. Returns a dictionary with arrays and metadata.
"""
function load_seismograms(filename::AbstractString)
    return npzread(filename)
end
