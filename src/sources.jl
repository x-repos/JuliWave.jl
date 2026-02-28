"""
    evaluate_wavelet(w::RickerWavelet, t) -> Float64

Ricker wavelet (second derivative of Gaussian):
    s(t) = A * (1 - 2π²f₀²(t-t₀)²) * exp(-π²f₀²(t-t₀)²)
"""
function evaluate_wavelet(w::RickerWavelet, t::Real)
    a = π^2 * w.f0^2
    tau = t - w.t0
    return w.amplitude * (1.0 - 2.0 * a * tau^2) * exp(-a * tau^2)
end

"""
    evaluate_wavelet(w::GaussianDerivativeWavelet, t) -> Float64

First derivative of Gaussian:
    s(t) = -A * 2π²f₀²(t-t₀) * exp(-π²f₀²(t-t₀)²)
"""
function evaluate_wavelet(w::GaussianDerivativeWavelet, t::Real)
    a = π^2 * w.f0^2
    tau = t - w.t0
    return -w.amplitude * 2.0 * a * tau * exp(-a * tau^2)
end

"""
    compute_source_timeseries(wavelet, nt, dt) -> Vector{Float64}

Precompute the full source time series for all time steps.
"""
function compute_source_timeseries(wavelet::AbstractWavelet, nt::Int, dt::Float64)
    return [evaluate_wavelet(wavelet, (it - 1) * dt) for it in 1:nt]
end
