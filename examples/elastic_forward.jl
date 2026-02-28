# Elastic Forward Modeling Example
# Run a 2D elastic simulation on a homogeneous model with C-PML boundaries

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliWave

# Grid setup
nx, ny = 401, 201
dx, dy = 10.0, 10.0  # meters
grid = Grid2D(nx, ny, dx, dy)

# Homogeneous elastic model
vp_val = 3300.0    # m/s
vs_val = vp_val / 1.732  # m/s
rho_val = 2800.0   # kg/m³
vp = fill(vp_val, nx, ny)
vs = fill(vs_val, nx, ny)
rho = fill(rho_val, nx, ny)
model = ElasticModel2D(vp, vs, rho, grid)

# Source: first derivative of Gaussian at off-center position
f0 = 7.0  # Hz
wavelet = GaussianDerivativeWavelet(f0; amplitude=1e7)
src = PointSource(2000.0, 700.0, wavelet; angle=135.0)

# Receivers
receivers = [Receiver(300.0, 300.0), Receiver(700.0, 300.0)]
geometry = Geometry([src], receivers)

# Time stepping
dt = suggest_dt(vp_val, dx, dy; courant_target=0.4)
nt = 500
config = SimulationConfig(nt, dt; pml_points=10)

println("Grid: $(nx) x $(ny), dx=$(dx) m")
println("Vp: $(vp_val) m/s, Vs: $(round(vs_val, digits=1)) m/s")
println("Source angle: 135° from Y axis")
println("Time steps: $(nt), dt=$(round(dt*1e3, digits=3)) ms")
println("Courant number: $(round(check_cfl(vp_val, dt, dx, dy), digits=3))")
println()

# Run simulation with wavefield snapshots
println("Running elastic forward simulation...")
seis_vx, seis_vy, snaps_vx, snaps_vy = simulate_elastic_wavefield(
    model, geometry, config; save_every=25)

println("Done!")
println("Seismogram size: $(size(seis_vx))")
println("Max |vx|: $(maximum(abs.(seis_vx)))")
println("Max |vy|: $(maximum(abs.(seis_vy)))")
println("Snapshots: $(size(snaps_vx, 3)) frames")

# Save wavefield snapshots as PPM images
outdir = joinpath(@__DIR__, "output", "elastic_forward")
mkpath(outdir)

function save_ppm(filename, field; power=0.3)
    ny_img, nx_img = size(field, 2), size(field, 1)
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

for i in 1:size(snaps_vx, 3)
    t_ms = round((i - 1) * 25 * dt * 1000, digits=1)
    save_ppm(joinpath(outdir, "vx_$(lpad(i, 3, '0')).ppm"), snaps_vx[:, :, i])
    save_ppm(joinpath(outdir, "vy_$(lpad(i, 3, '0')).ppm"), snaps_vy[:, :, i])
    println("  Saved snapshot $i (t=$(t_ms) ms)")
end

println("\nSnapshots saved to: $outdir")
println("View with: display $outdir/vx_*.ppm  (or use eog, feh, gimp)")
