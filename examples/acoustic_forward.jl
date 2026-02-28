# Acoustic Forward Modeling Example
# Run a 2D acoustic simulation on a homogeneous model with C-PML boundaries

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliWave

# Grid setup
nx, ny = 201, 201
dx, dy = 1.5, 1.5  # meters
grid = Grid2D(nx, ny, dx, dy)

# Homogeneous velocity model
vp_val = 2000.0   # m/s
rho_val = 2000.0   # kg/m³
vp = fill(vp_val, nx, ny)
rho = fill(rho_val, nx, ny)
model = AcousticModel2D(vp, rho, grid)

# Source: Ricker wavelet at center of model
f0 = 35.0  # Hz
src = PointSource(150.0, 150.0, RickerWavelet(f0))

# Receivers: line of receivers
receivers = [Receiver(50.0 + i * 10.0, 250.0) for i in 0:19]
geometry = Geometry([src], receivers)

# Time stepping
dt = suggest_dt(vp_val, dx, dy; courant_target=0.5)
nt = 500
config = SimulationConfig(nt, dt; pml_points=10)

println("Grid: $(nx) x $(ny), dx=$(dx) m")
println("Velocity: $(vp_val) m/s, Density: $(rho_val) kg/m³")
println("Source: Ricker f0=$(f0) Hz at (150, 150) m")
println("Time steps: $(nt), dt=$(round(dt*1e6, digits=1)) μs")
println("Courant number: $(round(check_cfl(vp_val, dt, dx, dy), digits=3))")
println()

# Run simulation with wavefield snapshots
println("Running acoustic forward simulation...")
seismograms, snapshots = simulate_acoustic_wavefield(model, geometry, config; save_every=50)

println("Done!")
println("Seismogram size: $(size(seismograms))")
println("Snapshot size: $(size(snapshots))")
println("Max pressure in seismograms: $(maximum(abs.(seismograms)))")
println("Number of snapshots: $(size(snapshots, 3))")

# Save wavefield snapshots as PPM images
outdir = joinpath(@__DIR__, "output", "acoustic_forward")
mkpath(outdir)

function save_ppm(filename, field; power=0.3)
    nx_img, ny_img = size(field)
    maxamp = maximum(abs.(field))
    maxamp == 0.0 && (maxamp = 1.0)
    open(filename, "w") do io
        println(io, "P3")
        println(io, "$nx_img $ny_img")
        println(io, "255")
        for jj in 1:ny_img
            for ii in 1:nx_img
                v = clamp(field[ii, jj] / maxamp, -1.0, 1.0)
                if abs(v) < 0.01
                    print(io, "255 255 255 ")
                elseif v >= 0
                    r = round(Int, 255 * v^power)
                    print(io, "$r 0 0 ")
                else
                    b = round(Int, 255 * abs(v)^power)
                    print(io, "0 0 $b ")
                end
            end
            println(io)
        end
    end
end

for i in 1:size(snapshots, 3)
    t_ms = round((i - 1) * 50 * dt * 1000, digits=1)
    save_ppm(joinpath(outdir, "pressure_$(lpad(i, 3, '0')).ppm"), snapshots[:, :, i])
    println("  Saved snapshot $i (t=$(t_ms) ms)")
end

println("\nSnapshots saved to: $outdir")
