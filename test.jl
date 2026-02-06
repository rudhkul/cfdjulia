using Printf
using Plots
using Interpolations
using LinearAlgebra
gr()

function plot_mesh(x, y, filename="lid_driven_mesh.png")
    # Quick sanity check to see if our grid stretching looks right
    p = scatter(vec(x), vec(y), markersize=1, aspect_ratio=:equal, legend=false, title="Mesh", xlims=(0,1), ylims=(0,1))
    savefig(p, filename)
end

function plot_fields(x, y, u, v, psi, vort, Re, ratiox, ratioy, filename=nothing)
    if filename === nothing
        filename = "lid_driven_fields_Re$(Int(Re))_ratio_$(ratiox).png"
    end
    
    # Just dumping the standard 4-panel plot: u, v, psi, and vorticity
    p1 = contourf(x[:,1], y[1,:], u', title="U Velocity", aspect_ratio=:equal, c=:viridis)
    p2 = contourf(x[:,1], y[1,:], v', title="V Velocity", aspect_ratio=:equal, c=:viridis)
    p3 = contourf(x[:,1], y[1,:], psi', title="Stream Function", aspect_ratio=:equal, c=:viridis)
    p4 = contourf(x[:,1], y[1,:], vort', title="Vorticity", aspect_ratio=:equal, c=:balance)

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800), title="Re=$(Int(Re))")
    savefig(fig, filename)
    println("Saved fields to $filename")
end

function plot_validation_vs_ghia(x, y, u, v, Re, ratiox, ratioy)
    ghia_x_v, ghia_v_ref, ghia_y_u, ghia_u_ref = ghia_data(Re)
    
    # We need the centerlines (vertical for U, horizontal for V)
    # finding the index closest to 0.5 since the grid might be stretched
    mid_x = argmin(abs.(x[:,1] .- 0.5))
    mid_y = argmin(abs.(y[1,:] .- 0.5))
    
    u_sim = u[mid_x, :]
    v_sim = v[:, mid_y]
    
    # Comparison plot 1: U velocity along the vertical centerline
    p1 = plot(ghia_u_ref, ghia_y_u, seriestype=:scatter, label="Ghia", title="U @ x=0.5", xlabel="u", ylabel="y")
    plot!(p1, u_sim, y[mid_x, :], linewidth=2, label="Sim")
    
    # Comparison plot 2: V velocity along the horizontal centerline
    p2 = plot(x[:, mid_y], v_sim, linewidth=2, label="Sim", title="V @ y=0.5", xlabel="x", ylabel="v")
    plot!(p2, ghia_x_v, ghia_v_ref, seriestype=:scatter, label="Ghia")
    
    fig = plot(p1, p2, layout=(1,2), size=(900, 400))
    savefig(fig, "validation_Re$(Int(Re)).png")
    
    # Interpolate our results to the exact Ghia points to get a clean error metric
    interp_u = LinearInterpolation(y[mid_x, :], u_sim, extrapolation_bc=Line())
    interp_v = LinearInterpolation(x[:, mid_y], v_sim, extrapolation_bc=Line())
    
    u_err = norm(interp_u(ghia_y_u) - ghia_u_ref) / sqrt(length(ghia_u_ref))
    v_err = norm(interp_v(ghia_x_v) - ghia_v_ref) / sqrt(length(ghia_v_ref))
    
    return u_err, v_err
end

function ghia_data(Re::Float64)
    # These are the classic benchmark numbers from Ghia et al (1982).
    # Don't touch these arrays unless you want to break the validation.
    ghia_y_u = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
    ghia_x_v = [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]

    if Re ≈ 100.0
        ghia_u = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]
        ghia_v = [0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0]
    elseif Re ≈ 400.0
        ghia_u = [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]
        ghia_v = [0.0, 0.18109, 0.19791, 0.21275, 0.23643, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.34273, -0.22847, -0.19254, -0.15663, -0.12146, 0.0]
    elseif Re ≈ 1000.0
        ghia_u = [0.0, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.0]
        ghia_v = [0.0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.0]
    else
        return ghia_data(100.0)
    end
    return ghia_x_v, ghia_v, ghia_y_u, ghia_u
end

function solve_lid_driven_cavity(M, N, ratiox, ratioy, Re, dt, crit_psi, crit_w, U_lid, use_adaptive_dt=true)
    
    x = zeros(M, N); y = zeros(M, N)
    
    # Building the grid. If ratio > 1, we squeeze points near the walls.
    # It's a bit verbose but necessary to handle the non-uniform spacing.
    if ratiox == 1.0
        for i in 1:M; x[i,:] .= (i-1)/(M-1); end
    else
        mid = (M+1)/2.0; Δ = 0.5*(1-ratiox)/(1-ratiox^(mid-1))
        for i in 1:Int(floor(mid)); x[i,:] .= Δ*(1-ratiox^(i-1))/(1-ratiox); end
        for i in Int(ceil(mid))+1:M; k=M+1-i; x[i,:] .= 1 - Δ*(1-ratiox^(k-1))/(1-ratiox); end
        if M%2!=0; x[Int(mid),:] .= 0.5; end
    end
    
    if ratioy == 1.0
        for j in 1:N; y[:,j] .= (j-1)/(N-1); end
    else
        mid = (N+1)/2.0; Δ = 0.5*(1-ratioy)/(1-ratioy^(mid-1))
        for j in 1:Int(floor(mid)); y[:,j] .= Δ*(1-ratioy^(j-1))/(1-ratioy); end
        for j in Int(ceil(mid))+1:N; k=N+1-j; y[:,j] .= 1 - Δ*(1-ratioy^(k-1))/(1-ratioy); end
        if N%2!=0; y[:,Int(mid)] .= 0.5; end
    end

    dx_vec = diff(x[:,1])
    dy_vec = diff(y[1,:])
    
    if use_adaptive_dt
        # Safety margin for CFL condition. 0.3 is conservative but safe.
        dt = 0.3 * minimum(dx_vec) / U_lid
        @printf("Adaptive dt set to: %.2e\n", dt)
    end
    
    ν = 1.0 / Re
    vort = zeros(M, N)
    psi  = zeros(M, N)
    
    # Start with a kick on the top lid or nothing happens
    vort[:,N] .= -2.0 * U_lid / dy_vec[end]
    
    iter = 0
    t = 0.0
    max_iters = 150000 
    
    vort_new = copy(vort)
    
    # Main time-stepping loop 
    while iter < max_iters
        iter += 1
        t += dt
        
        # Solving the Vorticity Transport equation
        # We have to be careful about diffusion vs advection here
        for i in 2:M-1, j in 2:N-1
            h_E = dx_vec[i]; h_W = dx_vec[i-1]
            h_N = dy_vec[j]; h_S = dy_vec[j-1]
            
            # Standard central difference for diffusion
            w_xx = 2.0 * (vort[i+1,j]*h_W + vort[i-1,j]*h_E - vort[i,j]*(h_E+h_W)) / (h_E*h_W*(h_E+h_W))
            w_yy = 2.0 * (vort[i,j+1]*h_S + vort[i,j-1]*h_N - vort[i,j]*(h_N+h_S)) / (h_N*h_S*(h_N+h_S))
            
            # Get local velocity from streamfunction
            u =  (psi[i,j+1] - psi[i,j-1]) / (h_N + h_S)
            v = -(psi[i+1,j] - psi[i-1,j]) / (h_E + h_W)
            
            # Advection: Switching to Second-Order Upwind (Linear Upwind)
            # This kills the false diffusion that was ruining the Re=1000 case.
            # If we are too close to the wall (i=2 or M-1), we fall back to 1st order to avoid indexing errors.
            
            # Handling X direction advection
            if u >= 0
                if i > 2
                    # The good stuff: using 2 points upstream (i, i-1, i-2)
                    dw_dx = (3*vort[i,j] - 4*vort[i-1,j] + vort[i-2,j]) / (2*h_W) 
                else
                    # Fallback near boundary
                    dw_dx = (vort[i,j] - vort[i-1,j]) / h_W
                end
            else
                if i < M-1
                    # 2 points upstream from the other side (i, i+1, i+2)
                    dw_dx = (-3*vort[i,j] + 4*vort[i+1,j] - vort[i+2,j]) / (2*h_E)
                else
                    dw_dx = (vort[i+1,j] - vort[i,j]) / h_E
                end
            end
            
            # Handling Y direction advection
            if v >= 0
                if j > 2
                    dw_dy = (3*vort[i,j] - 4*vort[i,j-1] + vort[i,j-2]) / (2*h_S)
                else
                    dw_dy = (vort[i,j] - vort[i,j-1]) / h_S
                end
            else
                if j < N-1
                    dw_dy = (-3*vort[i,j] + 4*vort[i,j+1] - vort[i,j+2]) / (2*h_N)
                else
                    dw_dy = (vort[i,j+1] - vort[i,j]) / h_N
                end
            end
            
            vort_new[i,j] = vort[i,j] + dt * (ν*(w_xx + w_yy) - (u*dw_dx + v*dw_dy))
        end
        
        vort .= vort_new
        
        # Solving Poisson equation for Psi using SOR
        # Omega=1.6 seems to be the sweet spot for stability here
        omega = 1.6
        max_err_psi = 0.0
        
        for _ in 1:100 
            psi_old_iter = copy(psi)
            for i in 2:M-1, j in 2:N-1
                h_E = dx_vec[i]; h_W = dx_vec[i-1]
                h_N = dy_vec[j]; h_S = dy_vec[j-1]
                
                Cx = 2.0 / (h_E * h_W * (h_E + h_W))
                Cy = 2.0 / (h_N * h_S * (h_N + h_S))
                denom = Cx*(h_E+h_W) + Cy*(h_N+h_S)
                
                psi_star = (Cx*(psi[i+1,j]*h_W + psi[i-1,j]*h_E) + 
                            Cy*(psi[i,j+1]*h_S + psi[i,j-1]*h_N) + vort[i,j]) / denom
                
                psi[i,j] = (1.0 - omega)*psi[i,j] + omega*psi_star
            end
            # If Psi stops changing, we're good for this time step
            if maximum(abs.(psi .- psi_old_iter)) < 1e-6
                break
            end
        end
        
        # Boundary Conditions (Thom's Formula)
        # CRITICAL: The negative sign is key here. The lid moves right, creating a clockwise vortex (negative).
        # Without this sign, the physics fights the math and everything blows up.
        
        # Top Wall (Moving Lid)
        h = dy_vec[end]
        vort[:,N] .= -2.0 .* (psi[:,N-1] .+ h .* U_lid) ./ (h^2)
        
        # Bottom Wall
        h = dy_vec[1]
        vort[:,1] .= -2.0 .* psi[:,2] ./ (h^2)
        
        # Left Wall
        h = dx_vec[1]
        vort[1,:] .= -2.0 .* psi[2,:] ./ (h^2)
        
        # Right Wall
        h = dx_vec[end]
        vort[M,:] .= -2.0 .* psi[M-1, :] ./ (h^2)
        
        if iter % 1000 == 0
            w_change = maximum(abs.(vort_new .- vort)) 
            @printf("Iter %d, t=%.3f, Center Psi=%.2e\n", iter, t, psi[Int(round(M/2)), Int(round(N/2))])
            
            # Simple "are we there yet?" check
            if iter > 10000 && w_change < 1e-8
                println("Converged.")
                break
            end
        end
    end
    
    # Calculate final velocities from the streamfunction
    u = zeros(M,N); v = zeros(M,N)
    for i in 2:M-1, j in 2:N-1
        u[i,j] =  (psi[i,j+1] - psi[i,j-1]) / (dy_vec[j] + dy_vec[j-1])
        v[i,j] = -(psi[i+1,j] - psi[i-1,j]) / (dx_vec[i] + dx_vec[i-1])
    end
    
    # Hardcode the lid velocity just to be sure
    u[:,N] .= U_lid
    
    return x, y, u, v, psi, vort
end

function run()
    M, N = 151, 151
    Re = 1000.0
    ratiox, ratioy = 1.0, 1.0 
    U = 1.0
    
    println("Running Lid Driven Cavity: Re=$Re, Grid=$M x $N")
    
    x, y, u, v, psi, vort = solve_lid_driven_cavity(M, N, ratiox, ratioy, Re, 0.001, 1e-6, 1e-6, U)
    
    plot_fields(x, y, u, v, psi, vort, Re, ratiox, ratioy)
    u_err, v_err = plot_validation_vs_ghia(x, y, u, v, Re, ratiox, ratioy)
    
    println("\nValidation Results (Re=$Re):")
    println("U-Velocity Error: $(round(u_err, digits=4))")
    println("V-Velocity Error: $(round(v_err, digits=4))")
end

run()
