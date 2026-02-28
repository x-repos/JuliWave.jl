# Acoustic FWI Example
# Full Waveform Inversion on a simple model using L-BFGS

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliWave

# Grid setup - small for demonstration
nx, ny = 41, 41
dx, dy = 10.0, 10.0
grid = Grid2D(nx, ny, dx, dy)

# True model: background + anomaly
vp_bg = 2000.0
rho_val = 2000.0
vp_true = fill(vp_bg, nx, ny)
vp_true[16:26, 16:26] .= 2300.0  # velocity anomaly in center
rho = fill(rho_val, nx, ny)
model_true = AcousticModel2D(vp_true, rho, grid)

# Source and receivers
f0 = 10.0
src = PointSource(50.0, 200.0, RickerWavelet(f0))
receivers = [Receiver(350.0, j * 40.0) for j in 1:9]
geometry = Geometry([src], receivers)

# Time stepping
dt = suggest_dt(2300.0, dx, dy; courant_target=0.3)
nt = 150
config = SimulationConfig(nt, dt; pml_points=5)

println("=== Acoustic FWI Example ===")
println("Grid: $(nx) x $(ny)")
println("True model: $(vp_bg) m/s background + 2300 m/s anomaly")
println()

# Generate observed data
println("Generating observed data with true model...")
observed = simulate_acoustic(model_true, geometry, config)
println("Max amplitude: $(maximum(abs.(observed)))")
println()

# Initial model (homogeneous)
vp_init = fill(vp_bg, nx, ny)
model_init = AcousticModel2D(vp_init, rho, grid)

println("Initial misfit:")
synthetic_init = simulate_acoustic(model_init, geometry, config)
misfit_init = l2_misfit(synthetic_init, observed)
println("  J = $(misfit_init)")
println()

# Run FWI (limited iterations for demonstration)
println("Running FWI with L-BFGS (3 iterations)...")
println("Note: This uses finite-difference gradients and is slow for large models.")
println()

model_opt, result = fwi(model_init, observed, geometry, config;
                        iterations=3,
                        vp_min=1500.0, vp_max=3000.0)

println()
println("=== Results ===")
println("Final misfit: $(Optim.minimum(result))")
println("Initial misfit: $(misfit_init)")
println("Misfit reduction: $(round((1 - Optim.minimum(result)/misfit_init)*100, digits=1))%")
println("Max recovered vp: $(maximum(model_opt.vp))")
println("Min recovered vp: $(minimum(model_opt.vp))")

# Save velocity models as PPM images
outdir = joinpath(@__DIR__, "output", "acoustic_fwi")
mkpath(outdir)

function save_ppm(filename, field; vmin=minimum(field), vmax=maximum(field))
    nx_img, ny_img = size(field)
    span = vmax - vmin
    span == 0.0 && (span = 1.0)
    open(filename, "w") do io
        println(io, "P3")
        println(io, "$nx_img $ny_img")
        println(io, "255")
        for jj in 1:ny_img
            for ii in 1:nx_img
                v = clamp((field[ii, jj] - vmin) / span, 0.0, 1.0)
                r = round(Int, 68 + 170 * v)
                g = round(Int, 1 + 254 * v * (1 - v) * 2)
                b = round(Int, 84 + 170 * (1 - v))
                print(io, "$r $g $b ")
            end
            println(io)
        end
    end
end

vmin, vmax = minimum(vp_true), maximum(vp_true)
save_ppm(joinpath(outdir, "vp_true.ppm"), vp_true; vmin, vmax)
save_ppm(joinpath(outdir, "vp_initial.ppm"), vp_init; vmin, vmax)
save_ppm(joinpath(outdir, "vp_recovered.ppm"), model_opt.vp; vmin, vmax)

println("\nSaved PPM images to: $outdir")
