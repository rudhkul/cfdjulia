using CUDA
using Printf
using Statistics

CUDA.allowscalar(false)

const D_CELLS = 20
const NX      = 61*D_CELLS + 1           # 1221
const NY      = 30*D_CELLS + 1           #  601
const TOTAL_N = NX * NY                  # 733 221

const CI0 = 10*D_CELLS + 1              # cylinder left   1-based  201
const CI1 = 11*D_CELLS + 1              # cylinder right           221
const CJ0 = 15*D_CELLS - D_CELLS÷2 + 1  # cylinder bottom         291
const CJ1 = 15*D_CELLS + D_CELLS÷2 + 1  # cylinder top            311
const CJ_MID = (CJ0 + CJ1) ÷ 2         # 301

const LX = 61.0f0;  const LY = 30.0f0
const DX = LX / Float32(NX - 1)         # 0.05000 D
const DY = LY / Float32(NY - 1)         # 0.04992 D

const RE          = 150.0f0
const U_INF       = 1.0f0
const T_END       = 400.0f0
const DT          = 0.016f0
const OMEGA_SOR   = 1.70f0
const N_SOR       = 300
const OUT_FREQ    = 25
const FLD_FREQ    = 4000
const PERTURB_AMP = 0.02f0   # v-perturbation amplitude to break symmetry

@inline function idx(i::Int32, j::Int32)::Int32
    return (j - Int32(1)) * Int32(NX) + i
end

@inline function is_cyl(i::Int32, j::Int32)::Bool
    return (i >= Int32(CI0)) & (i <= Int32(CI1)) &
           (j >= Int32(CJ0)) & (j <= Int32(CJ1))
end

function k_init!(u, v, p, fn_u, fn_v)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i > Int32(NX) || j > Int32(NY)) && return nothing
    id = idx(i, j)
    p[id] = fn_u[id] = fn_v[id] = 0.0f0
    v[id] = 0.0f0
    u[id] = is_cyl(i, j) ? 0.0f0 : U_INF
    return nothing
end

function k_perturb!(v)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i > Int32(NX) || j > Int32(NY)) && return nothing

    xlo = Int32(CI0) - Int32(2)
    xhi = Int32(CI1) + Int32(2)
    ylo = Int32(CJ0)            # ← was CJ0 - D_CELLS  (WRONG: 3-period sine)
    yhi = Int32(CJ1)            # ← was CJ1 + D_CELLS  (WRONG)

    if i >= xlo && i <= xhi && j >= ylo && j <= yhi && !is_cyl(i, j)
        arg = Float32(3.14159265f0) *
              Float32(j - Int32(CJ_MID)) / Float32(D_CELLS)
        # arg ∈ [−π/2, +π/2] → sin ∈ [−1, +1], monotone, no zero crossings
        v[idx(i, j)] += PERTURB_AMP * CUDA.sin(arg)
    end
    return nothing
end

function k_rhs!(u, v, fu, fv)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)

    if i < Int32(2) || i > Int32(NX)-Int32(1) ||
       j < Int32(2) || j > Int32(NY)-Int32(1) || is_cyl(i, j)
        if i <= Int32(NX) && j <= Int32(NY)
            id = idx(i,j); fu[id] = 0.0f0; fv[id] = 0.0f0
        end
        return nothing
    end

    id  = idx(i, j)
    uc  = u[id];  vc  = v[id]
    ue  = u[idx(i+Int32(1), j)];  uw = u[idx(i-Int32(1), j)]
    un  = u[idx(i, j+Int32(1))];  us_= u[idx(i, j-Int32(1))]
    ve  = v[idx(i+Int32(1), j)];  vw = v[idx(i-Int32(1), j)]
    vn  = v[idx(i, j+Int32(1))];  vs_= v[idx(i, j-Int32(1))]

    inv2dx = 0.5f0 / DX;  inv2dy = 0.5f0 / DY
    invdx2 = 1.0f0/(DX*DX);  invdy2 = 1.0f0/(DY*DY)

    conv_u = uc*(ue-uw)*inv2dx + vc*(un-us_)*inv2dy
    conv_v = uc*(ve-vw)*inv2dx + vc*(vn-vs_)*inv2dy
    diff_u = (ue-2.0f0*uc+uw)*invdx2 + (un-2.0f0*uc+us_)*invdy2
    diff_v = (ve-2.0f0*vc+vw)*invdx2 + (vn-2.0f0*vc+vs_)*invdy2

    fu[id] = -conv_u + diff_u * (1.0f0/RE)
    fv[id] = -conv_v + diff_v * (1.0f0/RE)
    return nothing
end

function k_pred!(u, v, fn_u, fn_v, fo_u, fo_v, us, vs, dt, c1::Float32, c2::Float32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i > Int32(NX) || j > Int32(NY)) && return nothing
    id = idx(i, j)

    if i==Int32(1)||i==Int32(NX)||j==Int32(1)||j==Int32(NY)||is_cyl(i,j)
        us[id] = u[id]; vs[id] = v[id]; return nothing
    end

    # c1,c2 are runtime Float32 values — branch is never constant-folded
    us[id] = u[id] + dt * (c1*fn_u[id] + c2*fo_u[id])
    vs[id] = v[id] + dt * (c1*fn_v[id] + c2*fo_v[id])
    return nothing
end

function k_bc_vel!(u, v)
    t = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)

    # bottom / top rows
    if t >= Int32(1) && t <= Int32(NX)
        u[idx(t, Int32(1))]  = u[idx(t, Int32(2))]
        v[idx(t, Int32(1))]  = 0.0f0
        u[idx(t, Int32(NY))] = u[idx(t, Int32(NY)-Int32(1))]
        v[idx(t, Int32(NY))] = 0.0f0
    end
    # outlet / inlet columns
    if t >= Int32(1) && t <= Int32(NY)
        u[idx(Int32(NX), t)] = u[idx(Int32(NX)-Int32(1), t)]
        v[idx(Int32(NX), t)] = v[idx(Int32(NX)-Int32(1), t)]
        u[idx(Int32(1),  t)] = U_INF
        v[idx(Int32(1),  t)] = 0.0f0
    end
    return nothing
end

function k_div!(us, vs, rhs, dt)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i > Int32(NX) || j > Int32(NY)) && return nothing
    id = idx(i, j)

    # zero on domain boundaries AND inside cylinder
    if i==Int32(1)||i==Int32(NX)||j==Int32(1)||j==Int32(NY)||is_cyl(i,j)
        rhs[id] = 0.0f0; return nothing
    end

    divu = (us[idx(i+Int32(1),j)]-us[idx(i-Int32(1),j)])*(0.5f0/DX) +
           (vs[idx(i,j+Int32(1))]-vs[idx(i,j-Int32(1))])*(0.5f0/DY)
    rhs[id] = divu / dt
    return nothing
end

function k_sor!(p, rhs, color::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i < Int32(2)||i > Int32(NX)-Int32(1)||
     j < Int32(2)||j > Int32(NY)-Int32(1)) && return nothing
    ((i + j) & Int32(1)) != color && return nothing

    id = idx(i, j)

    # cylinder interior: extrapolate pressure from fluid neighbours
    if is_cyl(i, j)
        s = 0.0f0; cnt = Int32(0)
        if !is_cyl(i+Int32(1),j) ; s+=p[idx(i+Int32(1),j)]; cnt+=Int32(1); end
        if !is_cyl(i-Int32(1),j) ; s+=p[idx(i-Int32(1),j)]; cnt+=Int32(1); end
        if !is_cyl(i,j+Int32(1)) ; s+=p[idx(i,j+Int32(1))]; cnt+=Int32(1); end
        if !is_cyl(i,j-Int32(1)) ; s+=p[idx(i,j-Int32(1))]; cnt+=Int32(1); end
        cnt > Int32(0) && (p[id] = s / Float32(cnt))
        return nothing
    end

    # Neumann ghost reflections at domain walls
    pe = p[idx(i+Int32(1), j)]
    pw = i > Int32(2)           ? p[idx(i-Int32(1),j)]   : p[idx(i+Int32(1),j)]
    pn = j < Int32(NY)-Int32(1) ? p[idx(i,j+Int32(1))]   : p[idx(i,j-Int32(1))]
    ps = j > Int32(2)           ? p[idx(i,j-Int32(1))]   : p[idx(i,j+Int32(1))]

    invdx2 = 1.0f0/(DX*DX);  invdy2 = 1.0f0/(DY*DY)
    coeff  = 2.0f0*(invdx2 + invdy2)
    p_gs   = ((pe+pw)*invdx2 + (pn+ps)*invdy2 - rhs[id]) / coeff
    p[id]  = (1.0f0 - OMEGA_SOR)*p[id] + OMEGA_SOR*p_gs
    return nothing
end

function k_bc_pres!(p)
    t = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if t >= Int32(1) && t <= Int32(NY)
        p[idx(Int32(NX), t)] = 0.0f0
        p[idx(Int32(1),  t)] = p[idx(Int32(2),  t)]
    end
    if t >= Int32(1) && t <= Int32(NX)
        p[idx(t, Int32(1))]  = p[idx(t, Int32(2))]
        p[idx(t, Int32(NY))] = p[idx(t, Int32(NY)-Int32(1))]
    end
    return nothing
end

function k_corr!(us, vs, p, u, v, dt)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i < Int32(2)||i > Int32(NX)-Int32(1)||
     j < Int32(2)||j > Int32(NY)-Int32(1)) && return nothing
    id = idx(i, j)
    if is_cyl(i, j); u[id]=0.0f0; v[id]=0.0f0; return nothing; end
    u[id] = us[id] - dt*(p[idx(i+Int32(1),j)]-p[idx(i-Int32(1),j)])*(0.5f0/DX)
    v[id] = vs[id] - dt*(p[idx(i,j+Int32(1))]-p[idx(i,j-Int32(1))])*(0.5f0/DY)
    return nothing
end

function k_vort!(u, v, w)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    (i < Int32(2)||i > Int32(NX)-Int32(1)||
     j < Int32(2)||j > Int32(NY)-Int32(1)) && return nothing
    id = idx(i, j)
    w[id] = (v[idx(i+Int32(1),j)]-v[idx(i-Int32(1),j)])*(0.5f0/DX) -
            (u[idx(i,j+Int32(1))]-u[idx(i,j-Int32(1))])*(0.5f0/DY)
    return nothing
end

const BLKX, BLKY = 16, 16
const GRDX  = cld(NX, BLKX)
const GRDY  = cld(NY, BLKY)
const BLK1  = 256
const GRD1  = cld(max(NX, NY), BLK1)

function compile_kernels(d_u, d_v, d_p, d_us, d_vs,
                          d_rhs, d_fn_u, d_fn_v, d_fo_u, d_fo_v, d_vd)
    kc_init    = @cuda launch=false k_init!(d_u, d_v, d_p, d_fn_u, d_fn_v)
    kc_perturb = @cuda launch=false k_perturb!(d_v)
    kc_rhs     = @cuda launch=false k_rhs!(d_u, d_v, d_fn_u, d_fn_v)
    kc_pred    = @cuda launch=false k_pred!(d_u, d_v, d_fn_u, d_fn_v,
                                             d_fo_u, d_fo_v, d_us, d_vs,
                                             DT, 1.0f0, 0.0f0)  # c1,c2 Float32 — no constant-folding
    kc_bc_vel  = @cuda launch=false k_bc_vel!(d_u, d_v)
    kc_div     = @cuda launch=false k_div!(d_us, d_vs, d_rhs, DT)
    kc_sor     = @cuda launch=false k_sor!(d_p, d_rhs, Int32(0))
    kc_bc_p    = @cuda launch=false k_bc_pres!(d_p)
    kc_corr    = @cuda launch=false k_corr!(d_us, d_vs, d_p, d_u, d_v, DT)
    kc_vort    = @cuda launch=false k_vort!(d_u, d_v, d_vd)
    return (kc_init, kc_perturb, kc_rhs, kc_pred,
            kc_bc_vel, kc_div, kc_sor, kc_bc_p, kc_corr, kc_vort)
end

function copy_fields!(hu, hv, hp, hw, d_u, d_v, d_p, d_vd, kc_vort)
    kc_vort(d_u, d_v, d_vd; threads=(BLKX,BLKY), blocks=(GRDX,GRDY))
    CUDA.synchronize()
    copyto!(hu, d_u); copyto!(hv, d_v)
    copyto!(hp, d_p); copyto!(hw, d_vd)
end

# Surface-integral force computation — fully Float32 throughout
function compute_forces(hu::Vector{Float32}, hv::Vector{Float32},
                         hp::Vector{Float32})
    Fx = 0.0f0; Fy = 0.0f0
    inv2DX = 0.5f0/DX;  inv2DY = 0.5f0/DY
    twoInvRE = 2.0f0/RE;  invRE = 1.0f0/RE
    @inline hid(i,j) = (j-1)*NX + i

    @inbounds for j in CJ0:CJ1
        pf   = hp[hid(CI0-1, j)]
        dudx = (hu[hid(CI0,j)]     - hu[hid(CI0-1,j)])   / DX
        dudy = (hu[hid(CI0-1,j+1)] - hu[hid(CI0-1,j-1)]) * inv2DY
        dvdx = (hv[hid(CI0,j)]     - hv[hid(CI0-1,j)])   / DX
        Fx  += ( pf - twoInvRE*dudx) * DY
        Fy  -= invRE*(dudy+dvdx) * DY
    end
    @inbounds for j in CJ0:CJ1
        pf   = hp[hid(CI1+1, j)]
        dudx = (hu[hid(CI1+1,j)]   - hu[hid(CI1,j)])     / DX
        dudy = (hu[hid(CI1+1,j+1)] - hu[hid(CI1+1,j-1)]) * inv2DY
        dvdx = (hv[hid(CI1+1,j)]   - hv[hid(CI1,j)])     / DX
        Fx  += (-pf + twoInvRE*dudx) * DY
        Fy  += invRE*(dudy+dvdx) * DY
    end
    @inbounds for i in CI0:CI1
        pf   = hp[hid(i, CJ0-1)]
        dvdy = (hv[hid(i,CJ0)]     - hv[hid(i,CJ0-1)])   / DY
        dudy = (hu[hid(i,CJ0)]     - hu[hid(i,CJ0-1)])   / DY
        dvdx = (hv[hid(i+1,CJ0-1)] - hv[hid(i-1,CJ0-1)]) * inv2DX
        Fx  -= invRE*(dudy+dvdx) * DX
        Fy  += ( pf - twoInvRE*dvdy) * DX
    end
    @inbounds for i in CI0:CI1
        pf   = hp[hid(i, CJ1+1)]
        dvdy = (hv[hid(i,CJ1+1)]   - hv[hid(i,CJ1)])     / DY
        dudy = (hu[hid(i,CJ1+1)]   - hu[hid(i,CJ1)])     / DY
        dvdx = (hv[hid(i+1,CJ1+1)] - hv[hid(i-1,CJ1+1)]) * inv2DX
        Fx  += invRE*(dudy+dvdx) * DX
        Fy  += (-pf + twoInvRE*dvdy) * DX
    end
    return 2.0f0*Fx, 2.0f0*Fy   # Cd, Cl
end

function write_tecplot(step, t, hu, hv, hp, hw)
    fname = @sprintf("field_%06d.dat", step)
    open(fname, "w") do fp
        @printf(fp,"TITLE = \"2D Flow Past Square Cylinder t=%.4f\"\n", t)
        @printf(fp,"VARIABLES = \"X\" \"Y\" \"U\" \"V\" \"P\" \"Vorticity\"\n")
        @printf(fp,"ZONE T=\"t=%.4f\", I=%d, J=%d, DATAPACKING=POINT\n",t,NX,NY)
        @inbounds for j in 1:NY, i in 1:NX
            id = (j-1)*NX + i
            @printf(fp,"%.6e %.6e %.6e %.6e %.6e %.6e\n",
                    Float32(i-1)*DX, Float32(j-1)*DY,
                    hu[id], hv[id], hp[id], hw[id])
        end
    end
    @printf("  [Tecplot] %s written\n", fname)
end

function write_cp(t, hp)
    D_len = Float32(CI1 - CI0) * DX    # 1.0 D
    q     = 0.5f0 * U_INF * U_INF
    @inline hid(i,j) = (j-1)*NX + i
    open("cp_cylinder.dat", "w") do fp
        @printf(fp,"# Cp around square cylinder  t=%.4f\n",t)
        @printf(fp,"# Cp = p_face/(0.5*U_inf^2), CCW from bottom-left corner\n")
        @printf(fp,"# s/D    Face    Cp\n")
        s = 0.0f0
        for j in CJ0:CJ1
            @printf(fp,"%.6e  L  %.6e\n",s/D_len, hp[hid(CI0-1,j)]/q); s+=DY; end
        for i in CI0:CI1
            @printf(fp,"%.6e  T  %.6e\n",s/D_len, hp[hid(i,CJ1+1)]/q); s+=DX; end
        for j in CJ1:-1:CJ0
            @printf(fp,"%.6e  R  %.6e\n",s/D_len, hp[hid(CI1+1,j)]/q); s+=DY; end
        for i in CI1:-1:CI0
            @printf(fp,"%.6e  B  %.6e\n",s/D_len, hp[hid(i,CJ0-1)]/q); s+=DX; end
    end
    @printf("  [Cp] cp_cylinder.dat written (4 D perimeter)\n")
end

function estimate_strouhal(th::Vector{Float32}, Clh::Vector{Float32})
    N = length(th);  N < 8 && return 0.0f0
    start = N ÷ 2;  cross_t = Float32[]
    for k in (start+1):N
        if Clh[k-1] < 0.0f0 && Clh[k] >= 0.0f0
            frac = -Clh[k-1]/(Clh[k]-Clh[k-1]+1.0f-30)
            push!(cross_t, th[k-1]+frac*(th[k]-th[k-1]))
        end
    end
    length(cross_t) < 2 && return 0.0f0
    T_avg = mean(diff(cross_t))
    return T_avg > 0.0f0 ? 1.0f0/T_avg : 0.0f0
end

function write_strouhal(St, th, Clh)
    open("strouhal.dat","w") do fp
        @printf(fp,"# Strouhal analysis  Estimated St = %.6f  (D=U_inf=1)\n",St)
        @printf(fp,"# Expected Re=150:  St ≈ 0.165 – 0.175\n#\n")
        @printf(fp,"# t           Cl\n")
        for k in eachindex(th); @printf(fp,"%.6e  %.6e\n",th[k],Clh[k]); end
    end
    @printf("  [St] strouhal.dat written  (St = %.4f)\n",St)
end

function main()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  2D Flow Past Square Cylinder — Julia/CUDA.jl FDM v2          ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    @printf("Grid     : %d × %d  (D_CELLS=%d,  Δx=%.4f D)\n", NX,NY,D_CELLS,DX)
    @printf("Re = %.1f  |  dt = %.5f  |  T_end = %.1f\n", RE, DT, T_END)
    @printf("Cylinder : x=[%d..%d] y=[%d..%d] centre=(10.5D, 15.0D)\n",
            CI0,CI1,CJ0,CJ1)
    @printf("CFL (conv) = %.4f   CFL (diff) = %.5f\n",
            U_INF*DT/DX, DT/(RE*DX*DX))
    @printf("SOR      : %d sweeps/step,  ω = %.2f\n", N_SOR, OMEGA_SOR)
    @printf("Perturb  : v += %.3f·sin(π·(j-CJ_mid)/D_CELLS)  [symmetry break]\n",
            PERTURB_AMP)
    @printf("GPU      : %s\n\n", CUDA.name(CUDA.device()))
    flush(stdout)

    d_u    = CUDA.zeros(Float32, TOTAL_N)
    d_v    = CUDA.zeros(Float32, TOTAL_N)
    d_p    = CUDA.zeros(Float32, TOTAL_N)
    d_us   = CUDA.zeros(Float32, TOTAL_N)
    d_vs   = CUDA.zeros(Float32, TOTAL_N)
    d_rhs  = CUDA.zeros(Float32, TOTAL_N)
    d_fn_u = CUDA.zeros(Float32, TOTAL_N)
    d_fn_v = CUDA.zeros(Float32, TOTAL_N)
    d_fo_u = CUDA.zeros(Float32, TOTAL_N)
    d_fo_v = CUDA.zeros(Float32, TOTAL_N)
    d_vd   = CUDA.zeros(Float32, TOTAL_N)

    hu = Vector{Float32}(undef, TOTAL_N)
    hv = Vector{Float32}(undef, TOTAL_N)
    hp = Vector{Float32}(undef, TOTAL_N)
    hw = Vector{Float32}(undef, TOTAL_N)

    print("Compiling GPU kernels... "); flush(stdout)
    t_jit = @elapsed begin
        (kc_init, kc_perturb, kc_rhs, kc_pred,
         kc_bc_vel, kc_div, kc_sor, kc_bc_p,
         kc_corr, kc_vort) = compile_kernels(
             d_u, d_v, d_p, d_us, d_vs,
             d_rhs, d_fn_u, d_fn_v, d_fo_u, d_fo_v, d_vd)
        CUDA.synchronize()
    end
    @printf("done  (%.1f s JIT)\n\n", t_jit)

    kc_init(d_u, d_v, d_p, d_fn_u, d_fn_v;
            threads=(BLKX,BLKY), blocks=(GRDX,GRDY))
    kc_perturb(d_v; threads=(BLKX,BLKY), blocks=(GRDX,GRDY))
    CUDA.synchronize()

    fp_force = open("force_time.dat","w")
    @printf(fp_force,"# 2D Flow Past Square Cylinder  Re=%.1f  (Julia/CUDA v2)\n",RE)
    @printf(fp_force,"# t           Cd           Cl\n")

    time_hist = Float32[]; Cl_hist = Float32[]
    sizehint!(time_hist, 4096); sizehint!(Cl_hist, 4096)

    N_STEPS    = round(Int, T_END / DT)
    ab_c1      = 1.0f0    # step 0: Euler  (1.0·Fⁿ + 0.0·Fⁿ⁻¹)
    ab_c2      = 0.0f0
    sim_time   = 0.0f0
    Cd_cur = 0.0f0; Cl_cur = 0.0f0

    @printf("Time loop: %d steps (%.1f D/U∞)\n\n", N_STEPS, T_END); flush(stdout)

    t_loop = @elapsed for step in 0:N_STEPS

        # diagnostics
        if step % OUT_FREQ == 0
            copy_fields!(hu, hv, hp, hw, d_u, d_v, d_p, d_vd, kc_vort)
            Cd_cur, Cl_cur = compute_forces(hu, hv, hp)
            @printf(fp_force,"%.6e  %.6e  %.6e\n", sim_time, Cd_cur, Cl_cur)
            flush(fp_force)
            push!(time_hist, sim_time); push!(Cl_hist, Cl_cur)
            @printf("  t=%8.3f/%6.1f  Cd=%7.4f  Cl=%+7.4f\n",
                    sim_time, T_END, Cd_cur, Cl_cur)
            flush(stdout)
        end

        if step % FLD_FREQ == 0
            step % OUT_FREQ != 0 &&
                copy_fields!(hu, hv, hp, hw, d_u, d_v, d_p, d_vd, kc_vort)
            write_tecplot(step, sim_time, hu, hv, hp, hw)
        end

        step == N_STEPS && break

        kc_rhs(d_u, d_v, d_fn_u, d_fn_v;
               threads=(BLKX,BLKY), blocks=(GRDX,GRDY))

        kc_pred(d_u, d_v, d_fn_u, d_fn_v, d_fo_u, d_fo_v,
                d_us, d_vs, DT, ab_c1, ab_c2;
                threads=(BLKX,BLKY), blocks=(GRDX,GRDY))

        kc_bc_vel(d_us, d_vs; threads=BLK1, blocks=GRD1)

        kc_div(d_us, d_vs, d_rhs, DT;
               threads=(BLKX,BLKY), blocks=(GRDX,GRDY))

        for s in 0:N_SOR-1
            kc_sor(d_p, d_rhs, Int32(s & 1);
                   threads=(BLKX,BLKY), blocks=(GRDX,GRDY))
            kc_bc_p(d_p; threads=BLK1, blocks=GRD1)
        end

        kc_corr(d_us, d_vs, d_p, d_u, d_v, DT;
                threads=(BLKX,BLKY), blocks=(GRDX,GRDY))

        kc_bc_vel(d_u, d_v; threads=BLK1, blocks=GRD1)

        d_fo_u, d_fn_u = d_fn_u, d_fo_u
        d_fo_v, d_fn_v = d_fn_v, d_fo_v

        ab_c1 = 1.5f0; ab_c2 = -0.5f0
        sim_time  += DT
    end

    CUDA.synchronize()
    @printf("\nWall time: %.1f s  (%.2f ms/step)\n",
            t_loop, 1000.0*t_loop/N_STEPS)

    # ── post-processing ────────────────────────────────────────────────────
    println("\n── Post-processing ──────────────────────────────────────")
    copy_fields!(hu, hv, hp, hw, d_u, d_v, d_p, d_vd, kc_vort)
    write_cp(sim_time, hp)
    St = estimate_strouhal(time_hist, Cl_hist)
    write_strouhal(St, time_hist, Cl_hist)
    close(fp_force)
    println("  [Force] force_time.dat written")

    println()
    println("╔══════════════════════════════════════════════════════════╗")
    @printf( "║  COMPLETE  t=%.2f D/U∞                                ║\n",sim_time)
    @printf( "║  Cd=%.4f  Cl=%+.4f  St≈%.4f                       ║\n",
             Cd_cur, Cl_cur, St)
    println( "╚══════════════════════════════════════════════════════════╝")
    return nothing
end

main()
