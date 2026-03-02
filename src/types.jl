struct Grid2D
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
end

abstract type AbstractWavelet end

struct RickerWavelet <: AbstractWavelet
    f0::Float64
    t0::Float64
    amplitude::Float64
end

RickerWavelet(f0; t0=1.2/f0, amplitude=1.0) = RickerWavelet(f0, t0, amplitude)

struct GaussianDerivativeWavelet <: AbstractWavelet
    f0::Float64
    t0::Float64
    amplitude::Float64
end

GaussianDerivativeWavelet(f0; t0=1.2/f0, amplitude=1.0) = GaussianDerivativeWavelet(f0, t0, amplitude)

struct PointSource{W<:AbstractWavelet}
    x::Float64
    y::Float64
    wavelet::W
    angle::Float64  # force angle in degrees from Y axis (elastic only)
end

PointSource(x, y, wavelet; angle=0.0) = PointSource(x, y, wavelet, angle)

struct Receiver
    x::Float64
    y::Float64
end

struct Geometry{S<:PointSource}
    sources::Vector{S}
    receivers::Vector{Receiver}
end

struct SimulationConfig
    nt::Int
    dt::Float64
    pml_points::Int
    pml_Rcoef::Float64
    pml_npower::Float64
    pml_kmax::Float64
    space_order::Int
    free_surface::Bool
end

function SimulationConfig(nt, dt; pml_points=10, pml_Rcoef=0.001, pml_npower=2.0, pml_kmax=1.0,
                          space_order=2, free_surface=false)
    SimulationConfig(nt, dt, pml_points, pml_Rcoef, pml_npower, pml_kmax, space_order, free_surface)
end
