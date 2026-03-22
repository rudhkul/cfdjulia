# 2D Flow Past Square Cylinder (CUDA.jl)

This repository contains a 2D incompressible Navier-Stokes solver for flow past a square cylinder at $Re = 150$. It is implemented in Julia and fully executed on the GPU using `CUDA.jl`.

## Methodology
The code employs a standard fractional step (projection) Finite Difference Method on a uniform Cartesian grid ($1221 \times 601$). 
* **Convection/Diffusion:** Adams-Bashforth scheme.
* **Pressure Poisson Equation:** Successive Over-Relaxation (SOR). 
* **Symmetry Breaking:** An initial sinusoidal perturbation is applied to the $v$-velocity field to accelerate the onset of vortex shedding.

*Note: The use of Red-Black SOR for the pressure Poisson solver is algorithmically obsolete and computationally inefficient compared to Krylov subspace or Multigrid methods, but it is sufficient for a toy problem of this resolution.*

## Requirements
* Julia >= 1.8
* NVIDIA GPU
* Packages: `CUDA`, `Printf`, `Statistics`

## Usage
Execute the script from the command line:
`julia main.jl`

## Outputs
The simulation generates the following files:
* `field_xxxxxx.dat`: Full field variables ($u, v, p, \omega$) in Tecplot format.
* `force_time.dat`: Time history of Drag ($C_d$) and Lift ($C_l$) coefficients.
* `cp_cylinder.dat`: Mean surface pressure coefficient ($C_p$) mapping.
* `strouhal.dat`: Automated Strouhal ($St$) number estimation from zero-crossings of the lift coefficient.
