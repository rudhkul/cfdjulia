# CFD in Julia

Two small incompressible-flow solvers in Julia:

* `test.jl`: lid-driven cavity at $Re = 1000$ on the CPU, validated against Ghia et al. (1982).
* `flow_past_square_cylinder_2d.jl`: flow past a square cylinder at $Re = 150$ on the GPU with `CUDA.jl`.

# Lid-Driven Cavity (`test.jl`)

A streamfunction–vorticity solver for the classic lid-driven cavity on a $151 \times 151$ grid.

* **Vorticity transport:** explicit time stepping, second-order upwind advection (first-order next to walls), central-difference diffusion.
* **Streamfunction Poisson equation:** SOR with $\omega = 1.6$.
* **Wall vorticity:** Thom's formula.
* **Validation:** centreline $u$ and $v$ compared with Ghia, Ghia and Shin, *J. Comput. Phys.* 48, 387–411 (1982).

## Requirements
* Julia 1.x
* Packages: `Plots`, `Interpolations` (plus the standard libraries `Printf` and `LinearAlgebra`)

## Usage
`julia test.jl`

## Outputs
* `lid_driven_fields_Re1000_ratio_1.0.png`: $u$, $v$, streamfunction and vorticity.
* `validation_Re1000.png`: centreline velocities against Ghia et al.
* The $u$ and $v$ errors against the reference data, printed to the terminal.

# 2D Flow Past Square Cylinder (CUDA.jl, `flow_past_square_cylinder_2d.jl`)

This script is a 2D incompressible Navier-Stokes solver for flow past a square cylinder at $Re = 150$. It is implemented in Julia and fully executed on the GPU using `CUDA.jl`.

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
`julia flow_past_square_cylinder_2d.jl`

## Outputs
The simulation generates the following files:
* `field_xxxxxx.dat`: Full field variables ($u, v, p, \omega$) in Tecplot format.
* `force_time.dat`: Time history of Drag ($C_d$) and Lift ($C_l$) coefficients.
* `cp_cylinder.dat`: Mean surface pressure coefficient ($C_p$) mapping.
* `strouhal.dat`: Automated Strouhal ($St$) number estimation from zero-crossings of the lift coefficient.

## License

MIT. See [LICENSE](LICENSE).
