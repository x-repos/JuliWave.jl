# Marmousi II Elastic Forward Modeling
# Matching DENISE-Black-Edition pyapi_demo.ipynb configuration
#
# DENISE config:
#   NX=500, NY=174, DH=20m
#   DT=2.0e-3, TIME=6.0s (NT=3000)
#   Ricker wavelet f0=10 Hz
#   PML: 10 gridpoints, npower=4, k_max=1.0
#   Sources: x=800-8720m, depth=40m, spacing=160m
#   Receivers: x=800-8780m, depth=460m, spacing=20m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliWave

# ── Load Marmousi II binary model files ──────────────────────────
denise_dir = "/home/x/Programs/denise/DENISE-Black-Edition-master/par/start"

function load_denise_binary(path, nx_file, ny_file)
    raw = Vector{Float32}(undef, nx_file * ny_file)
    open(path, "r") do io
        read!(io, raw)
    end
    # Binary is stored in C-order (row-major): reshape to (ny, nx) in Julia,
    # then transpose to get (nx, ny) matching DENISE grid
    return Float64.(permutedims(reshape(raw, ny_file, nx_file), (2, 1)))
end

NX_FILE, NY_FILE = 500, 174
vp = load_denise_binary(joinpath(denise_dir, "marmousi_II_marine.vp"), NX_FILE, NY_FILE)
vs = load_denise_binary(joinpath(denise_dir, "marmousi_II_marine.vs"), NX_FILE, NY_FILE)
rho = load_denise_binary(joinpath(denise_dir, "marmousi_II_marine.rho"), NX_FILE, NY_FILE)

println("Model loaded: Marmousi II Marine")
println("  Vp range:  $(minimum(vp)) - $(maximum(vp)) m/s")
println("  Vs range:  $(minimum(vs)) - $(maximum(vs)) m/s")
println("  Rho range: $(minimum(rho)) - $(maximum(rho)) kg/m³")

# Clamp vs to small positive value to avoid numerical issues at water/solid boundary
vs .= max.(vs, 1.0)

# ── Grid ─────────────────────────────────────────────────────────
dx = 20.0
nx, ny = NX_FILE, NY_FILE
grid = Grid2D(nx, ny, dx, dx)
model = ElasticModel2D(vp, vs, rho, grid)

println("\nGrid: $(nx) x $(ny), dx=$(dx) m")
println("Domain: $((nx-1)*dx) x $((ny-1)*dx) m")

# ── Acquisition (matching DENISE notebook) ───────────────────────
f0 = 10.0  # Hz (from DENISE source file)

# Sources: x=800-8720m, spacing 160m, depth 40m
src_x = collect(800.0:160.0:8720.0)
src_depth = 40.0
println("\nSources: $(length(src_x)) shots, x=$(src_x[1])-$(src_x[end]) m, depth=$(src_depth) m")

# Receivers: x=800-8780m, spacing 20m, depth 460m
rec_x = collect(800.0:20.0:8780.0)
rec_depth = 460.0
receivers = [Receiver(x, rec_depth) for x in rec_x]
println("Receivers: $(length(receivers)), x=$(rec_x[1])-$(rec_x[end]) m, depth=$(rec_depth) m")

# ── Simulation config (matching DENISE) ──────────────────────────
dt = 2.0e-3   # 2 ms (same as DENISE)
nt = 3000      # TIME=6.0s / DT=2e-3
config = SimulationConfig(nt, dt; pml_points=10, pml_npower=4.0, pml_kmax=1.0, space_order=8)

courant = check_cfl(maximum(vp), dt, dx, dx; space_order=8)
println("\ndt=$(dt*1e3) ms, nt=$(nt), T=$(nt*dt) s")
println("Courant number: $(round(courant, digits=4))")

# ── Run single shot (first source) ──────────────────────────────
shot_idx = 1
wavelet = RickerWavelet(f0; amplitude=1.0)
src = PointSource(src_x[shot_idx], src_depth, wavelet; angle=0.0)
geometry = Geometry([src], receivers)

snap_interval = 150  # save snapshot every 0.3s

println("\nRunning elastic forward for shot $(shot_idx) at x=$(src_x[shot_idx]) m ...")
println("  ($(nx)x$(ny) grid, $(nt) time steps, snapshot every $(snap_interval) steps)")

seis_vx, seis_vy, snaps_vx, snaps_vy = simulate_elastic_wavefield(
    model, geometry, config; save_every=snap_interval)

println("Done!")
println("Max |vx| seismogram: $(maximum(abs.(seis_vx)))")
println("Max |vy| seismogram: $(maximum(abs.(seis_vy)))")
println("Snapshots: $(size(snaps_vx, 3)) frames")

# ── Save output ──────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "output", "marmousi_elastic_forward")
mkpath(outdir)

function save_ppm(filename, field; power=0.3)
    nx_img, ny_img = size(field)
    maxamp = maximum(abs.(field))
    maxamp == 0.0 && (maxamp = 1.0)
    open(filename, "w") do io
        println(io, "P3")
        println(io, "$nx_img $ny_img")
        println(io, "255")
        for jj in ny_img:-1:1
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
    t_ms = round((i - 1) * snap_interval * dt * 1000, digits=1)
    save_ppm(joinpath(outdir, "vx_$(lpad(i, 3, '0')).ppm"), snaps_vx[:, :, i])
    save_ppm(joinpath(outdir, "vy_$(lpad(i, 3, '0')).ppm"), snaps_vy[:, :, i])
    println("  Saved snapshot $i (t=$(t_ms) ms)")
end

# Save seismograms as binary (nt x nrec, Float64)
open(joinpath(outdir, "seismogram_vx.bin"), "w") do io
    write(io, seis_vx)
end
open(joinpath(outdir, "seismogram_vy.bin"), "w") do io
    write(io, seis_vy)
end

println("\nOutput saved to: $outdir")
println("  Wavefield snapshots: vx_*.ppm, vy_*.ppm")
println("  Seismograms: seismogram_vx.bin, seismogram_vy.bin ($(nt)x$(length(receivers)) Float64)")
