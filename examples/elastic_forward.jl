# Elastic Forward Modeling Example
# Run a 2D elastic simulation on a homogeneous model with C-PML boundaries

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliWave

# Grid setup
nx, ny = 101, 101
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
src = PointSource(500.0, 700.0, wavelet; angle=135.0)

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

# Run simulation
println("Running elastic forward simulation...")
seis_vx, seis_vy = simulate_elastic(model, geometry, config)

println("Done!")
println("Seismogram size: $(size(seis_vx))")
println("Max |vx|: $(maximum(abs.(seis_vx)))")
println("Max |vy|: $(maximum(abs.(seis_vy)))")
