module JuliWave

using LinearAlgebra

# Core types
include("types.jl")
include("grid.jl")
include("sources.jl")
include("receivers.jl")
include("cpml.jl")
include("fd_coefficients.jl")
include("utils.jl")

# Acoustic solver
include("acoustic/acoustic_types.jl")
include("acoustic/acoustic_kernels.jl")
include("acoustic/acoustic_forward.jl")

# Elastic solver
include("elastic/elastic_types.jl")
include("elastic/elastic_kernels.jl")
include("elastic/elastic_forward.jl")

# FWI
include("fwi.jl")

# Exports - Types
export Grid2D, SimulationConfig
export AbstractWavelet, RickerWavelet, GaussianDerivativeWavelet
export PointSource, Receiver, Geometry
export AcousticModel2D, AcousticState2D
export ElasticModel2D, ElasticState2D
export CPMLCoefficients1D, CPML2D

# Exports - Grid
export snap_to_grid, grid_coords

# Exports - Sources/Receivers
export evaluate_wavelet, compute_source_timeseries
export snap_receivers, record_receivers!

# Exports - CPML
export setup_cpml

# Exports - Solvers
export simulate_acoustic, simulate_acoustic_wavefield
export simulate_elastic, simulate_elastic_wavefield

# Exports - FWI
export l2_misfit, compute_gradient_fd, fwi

# Exports - FD coefficients
export fd_coefficients, fd_stencil_x, fd_stencil_y

# Exports - Utils
export check_cfl, suggest_dt, select_backend

end # module JuliWave
