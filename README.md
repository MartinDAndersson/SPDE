# SPDE - Stochastic Partial Differential Equations

This project implements numerical methods for solving and analyzing stochastic partial differential equations (SPDEs), with a focus on estimating diffusion coefficients from observed solution paths.

## Overview

The SPDE package provides tools for:

1. Numerically solving stochastic heat equations with various diffusion coefficients
2. Analyzing solution paths using partial integration methods
3. Recovering the underlying diffusion coefficient (σ²) using machine learning techniques
4. Exploring how estimation accuracy scales with various discretization parameters

## Installation

To reproduce this project:

1. Clone the repository
2. Open a Julia console and run:
```julia
using Pkg
Pkg.add("DrWatson")  # Install globally for using `quickactivate`
Pkg.activate("path/to/this/project")
Pkg.instantiate()
```

This will install all necessary dependencies.

## Key Components

### Core Modules

- **SPDE.jl**: Main module with functions for solving SPDEs and estimating diffusion coefficients
- **gpu.jl**: GPU-accelerated implementation using CUDA
- **highdim.jl**: Tools for handling higher-dimensional problems (2D SPDEs)

### Key Functions

- `generate_solution(σ, nx, nt)`: Generate numerical solutions to the SPDE with diffusion coefficient σ
- `L_op(u, dt, dx)`: Apply the differential operator to extract information about σ²
- `partial_integration(solution, dt, dx, x_quot, t_quot, eps)`: Perform partial integration analysis
- `train_tuned(df)`: Train machine learning models to estimate σ² from solution data

## Mathematical Background

We consider stochastic heat equations of the form:

```
∂u/∂t = (1/2)∂²u/∂x² + σ(u)∂W/∂t
```

where W is a cylindrical Wiener process. The project investigates methods to recover the function σ² by analyzing solution paths.

## Experiments

The `notebooks/` directory contains Pluto notebooks exploring various aspects of the SPDE solutions:

- Parameter studies on discretization (dx, dt) effects
- Convergence analysis of the estimation methods
- Visualization of solutions and estimated diffusion coefficients
- GPU acceleration for 2D problems

## Getting Started

After installation, you can:

1. Explore the notebooks in the `notebooks/` directory
2. Run experiments from the `scripts/` directory
3. Use the core functionality by importing the SPDE module:

```julia
using DrWatson
@quickactivate "SPDE"
using SPDE

# Generate a solution with a sine diffusion coefficient
σ(x) = sin(x)
solution = SPDE.generate_solution(σ, 2^8, 2^16)

# Analyze the solution
df = SPDE.partial_integration(solution, dt, dx, 2, 2, 4*dx)

# Train a model to recover σ²
model = SPDE.train_tuned(df)
```

## Dependencies

Main dependencies include:
- DrWatson (scientific project management)
- DifferentialEquations.jl (SDE solvers)
- MLJ (machine learning)
- CUDA.jl (GPU acceleration)
- Plots.jl and CairoMakie (visualization)

## Citation

If you use this code in your research, please cite:
[Add citation information when availabl

