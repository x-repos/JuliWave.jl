using Optim

"""
    l2_misfit(synthetic, observed) -> Float64

Compute L2 waveform misfit: 0.5 * sum((synthetic - observed)²)
"""
function l2_misfit(synthetic::AbstractMatrix, observed::AbstractMatrix)
    return 0.5 * sum((synthetic .- observed) .^ 2)
end

"""
    acoustic_objective(vp_vec, model_template, observed_data, geometry, config)

Compute the L2 misfit for a given velocity model (as a vector).
Used internally by the FWI optimization.
"""
function acoustic_objective(vp_vec::AbstractVector, model_template::AcousticModel2D,
                            observed_data::AbstractMatrix, geometry::Geometry,
                            config::SimulationConfig)
    grid = model_template.grid
    vp = reshape(vp_vec, grid.nx, grid.ny)
    model = AcousticModel2D(vp, model_template.rho, grid)
    synthetic = simulate_acoustic(model, geometry, config)
    return l2_misfit(synthetic, observed_data)
end

"""
    compute_gradient_fd(model, observed_data, geometry, config; h=1.0)

Compute gradient of L2 misfit w.r.t. vp using finite differences.
This is a reference implementation for validating AD gradients.
"""
function compute_gradient_fd(model::AcousticModel2D, observed_data::AbstractMatrix,
                             geometry::Geometry, config::SimulationConfig; h::Float64=1.0)
    grid = model.grid
    vp0 = copy(model.vp)
    grad = zeros(Float64, grid.nx, grid.ny)

    # Base misfit
    syn0 = simulate_acoustic(model, geometry, config)
    J0 = l2_misfit(syn0, observed_data)

    for j in 1:grid.ny
        for i in 1:grid.nx
            vp_pert = copy(vp0)
            vp_pert[i, j] += h
            model_pert = AcousticModel2D(vp_pert, model.rho, grid)
            syn_pert = simulate_acoustic(model_pert, geometry, config)
            J_pert = l2_misfit(syn_pert, observed_data)
            grad[i, j] = (J_pert - J0) / h
        end
    end

    return grad
end

"""
    fwi(model_init, observed_data, geometry, config; method=LBFGS(), iterations=20,
        vp_min=nothing, vp_max=nothing)

Run Full Waveform Inversion using Optim.jl.

Uses finite-difference gradients by default. For Enzyme AD gradients,
use `compute_gradient_enzyme` when Enzyme support is available.

Returns (optimized_model, optimization_result).
"""
function fwi(model_init::AcousticModel2D, observed_data::AbstractMatrix,
             geometry::Geometry, config::SimulationConfig;
             method=Optim.LBFGS(), iterations::Int=20,
             vp_min::Union{Nothing,Float64}=nothing,
             vp_max::Union{Nothing,Float64}=nothing)
    grid = model_init.grid
    x0 = vec(copy(model_init.vp))

    function fg!(F, G, x)
        vp = reshape(x, grid.nx, grid.ny)
        model = AcousticModel2D(vp, model_init.rho, grid)
        synthetic = simulate_acoustic(model, geometry, config)

        if G !== nothing
            # Finite-difference gradient (per-parameter perturbation)
            h = 1.0
            J0 = l2_misfit(synthetic, observed_data)
            for idx in eachindex(x)
                x_pert = copy(x)
                x_pert[idx] += h
                vp_pert = reshape(x_pert, grid.nx, grid.ny)
                model_pert = AcousticModel2D(vp_pert, model_init.rho, grid)
                syn_pert = simulate_acoustic(model_pert, geometry, config)
                G[idx] = (l2_misfit(syn_pert, observed_data) - J0) / h
            end
            if F !== nothing
                return J0
            end
        end

        if F !== nothing
            return l2_misfit(synthetic, observed_data)
        end
    end

    if vp_min !== nothing && vp_max !== nothing
        lower = fill(vp_min, length(x0))
        upper = fill(vp_max, length(x0))
        result = Optim.optimize(Optim.only_fg!(fg!), lower, upper, x0,
                                Optim.Fminbox(method),
                                Optim.Options(iterations=iterations, show_trace=true))
    else
        result = Optim.optimize(Optim.only_fg!(fg!), x0, method,
                                Optim.Options(iterations=iterations, show_trace=true))
    end

    vp_opt = reshape(Optim.minimizer(result), grid.nx, grid.ny)
    model_opt = AcousticModel2D(vp_opt, model_init.rho, grid)

    return model_opt, result
end
