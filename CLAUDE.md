# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JuliWave.jl is a Julia package for 2D seismic wave simulation implementing acoustic and elastic wave equation solvers, C-PML absorbing boundaries, and Full Waveform Inversion (FWI). It is translated from Fortran reference implementations (Komatitsch et al.'s seismic_CPML codes).

## Common Commands

```bash
# Activate and install dependencies
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project test/test_acoustic.jl

# Run an example
julia --project examples/acoustic_forward.jl

# Interactive REPL with project activated
julia --project
```

## Architecture

The package uses a modular finite-difference solver architecture separated by physics type:

**Core layer** (`src/types.jl`, `src/grid.jl`, `src/sources.jl`, `src/receivers.jl`, `src/cpml.jl`, `src/fd_coefficients.jl`, `src/utils.jl`): Shared types (Grid2D, SimulationConfig, wavelets, geometry), C-PML boundary coefficient computation, and staggered-grid finite-difference coefficients via the Fornberg formula (orders 2–16).

**Acoustic solver** (`src/acoustic/`): Scalar pressure formulation. `acoustic_types.jl` defines AcousticModel2D/AcousticState2D, `acoustic_kernels.jl` has stencil operations with CPML, `acoustic_forward.jl` runs the time-stepping loop.

**Elastic solver** (`src/elastic/`): Velocity-stress formulation (vx, vy, σxx, σyy, σxy) for 2D isotropic elasticity. Same structure as acoustic: types, kernels, forward solver.

**FWI** (`src/fwi.jl`): L2 misfit, finite-difference gradient estimation, and Optim.jl-based LBFGS inversion.

**I/O** (`src/io.jl`): NPZ file I/O for saving/loading seismograms and arrays via the NPZ package.

**Key design decisions:**
- Staggered grid (half-point) finite differences for spatial derivatives
- Explicit time stepping with CFL stability constraint
- C-PML memory variables stored in state objects for boundary absorption
- Backend-agnostic arrays — `select_backend(use_gpu)` switches between CPU Array and CUDA CuArray
- Ghost cell padding replicates boundary values for higher-order FD stencils (pad = half_order - 1)
- Source injection: `src_type=:force` (angle-dependent force into vx/vy) or `src_type=:pressure` (explosive source into σ_xx/σ_yy)
- Free surface via stress imaging: σ_yy=0 at surface + antisymmetric mirror into ghost cells
- PML disabled at top boundary when free surface is active

## Source injection for cross-package comparison

When comparing with SPECFEM2D or Devito using pressure sources: in the velocity-stress formulation, injecting S(t) into σ_xx/σ_yy produces an effective source S'(t) in the pressure wave equation. To get a Ricker wavelet in pressure seismograms, inject `GaussianDerivativeWavelet` (not `RickerWavelet`). Devito handles this by loading SPECFEM's exact STF and differentiating it.

## Dependencies

Key: CUDA (GPU), Enzyme (AD), Optim (optimization), NPZ (file I/O), PyPlot (plotting). Requires Julia 1.10+.
