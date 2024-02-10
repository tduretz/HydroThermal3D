const USE_GPU  = false
const GPU_ID   = 0
const USE_MPI  = false

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
    macro sqrt(args...) esc(:(CUDA.sqrt($(args...)))) end
    macro exp(args...)  esc(:(CUDA.exp($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 3)
    macro sqrt(args...) esc(:(Base.sqrt($(args...)))) end
    macro exp(args...)  esc(:(Base.exp($(args...)))) end
end

using Printf, Statistics, LinearAlgebra, Plots
# using HDF5
using WriteVTK
# plotlyjs()
gr()

include("./Macros.jl")  # Include macros - Cachemisère
# include("./Weno5_Routines.jl")

include("./Weno5_Routines_v2.jl")

##########################################

@views function AdvectWithWeno5_v2( Tc_ex, Tc_exxx, Told, Vx, Vy, Vz, dt, _dx, _dy, _dz, Ttop, Tbot )

    @printf("Advecting with Weno5!\n")
    # Advection
    order = 2.0

    # Boundaries
    BC_type_W = 0
    BC_val_W  = 0.0
    BC_type_E = 0
    BC_val_E  = 0.0

    BC_type_S = 1
    BC_val_S  = Tbot
    BC_type_N = 1
    BC_val_N  = Ttop

    BC_type_B = 0
    BC_val_B  = 0.0
    BC_type_F = 0
    BC_val_F  = 0.0

    ########

    # Advect in x direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_x_Weno5!(Tc_exxx, Tc_ex, BC_type_W, BC_val_W, BC_type_E, BC_val_E)
        @parallel Advect!(Tc_ex, Tc_exxx, Vx, 1, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    # Advect in y direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_y_Weno5!(Tc_exxx, Tc_ex, BC_type_S, BC_val_S, BC_type_N, BC_val_N)
        @parallel Advect!(Tc_ex, Tc_exxx, Vy, 2, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    # Advect in z direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_z_Weno5!(Tc_exxx, Tc_ex, BC_type_B, BC_val_B, BC_type_F, BC_val_F)
        @parallel Advect!(Tc_ex, Tc_exxx, Vz, 3, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    ####
    @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum(Tc_ex), maximum(Tc_ex) )
end

##########################################

function TestAdvection()

    nt       = 50
    fact     = 16
    ncx      = fact*8 -6
    ncy      = fact*8 -6
    ncz      = 3#fact*32-6
    C        = 0.5

    xmin     =  -1.0;  xmax =    1.;   Lx = xmax - xmin
    ymin     =  -1.0;  ymax =    1.;   Ly = ymax - ymin
    zmin     = -0.00;  zmax =   1.0;   Lz = zmax - zmin

    # Preprocessing
    if (USE_MPI) me, dims, nprocs, coords, comm = init_global_grid(ncx, ncy, ncz; dimx=2, dimy=2, dimz=2);             
    else         me, dims, nprocs, coords       = (0, [1,1,1], 1, [0,0,0]);
    end
    Nix      = USE_MPI ? nx_g() : ncx                                                
    Niy      = USE_MPI ? ny_g() : ncy                                                
    Niz      = USE_MPI ? nz_g() : ncz                                                
    dx, dy, dz = Lx/Nix, Ly/Niy, Lz/Niz                                            
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz

    Vx       = @zeros(ncx+1,ncy  ,ncz  ) # Solution array 
    Vy       = @zeros(ncx  ,ncy+1,ncz  ) # Solution array 
    Vz       = @zeros(ncx  ,ncy  ,ncz+1) # Solution array 
    Tc_ex    = @zeros(ncx+2,ncy+2,ncz+2) # Solution array

    Vxm      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Vxp      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Vym      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Vyp      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Vzm      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Vzp      = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    v1       = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2 
    v2       = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2 
    v3       = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2 
    v4       = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2 
    v5       = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2 
    dTdxp    = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    dTdxm    = @zeros(ncx+0,ncy+0,ncz+0) # Not needed in v2
    Tc       = @zeros(ncx+0,ncy+0,ncz+0)
    Tc_exxx  = @zeros(ncx+6,ncy+6,ncz+6)

    xce = LinRange(xmin-dx/2, xmax+dx/2, ncx+2)
    yce = LinRange(ymin-dy/2, ymax+dy/2, ncy+2)
    zce = LinRange(xmin-dz/2, zmax+dz/2, ncz+2)

    Xc0 = @zeros(ncx ,ncy  ,ncz  )
    Tc  = @zeros(ncx ,ncy  ,ncz  )

    sx, sy = 0.1, 0.1
    x0, y0 = -Lx/4, -Ly/4
    # x0, y0 =  Lx/4,  Ly/4
    for i=1:ncx+2, j=1:ncy+2, k=1:ncz+2
        Tc_ex[i,j,k] = 5*exp( -((xce[i]-x0).^2)/sx^2 -((yce[j]-y0).^2)/sy^2 ) + 1.0
    end
    Ttop, Tbot = 1.0, 1.0 

    time = 0.
    Vx  .= 1.0
    Vy  .= 1.0

    dt   = C*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
    _dt  = 1.0/dt

    ## Action
    for it = 1:nt

        time += dt

        # Original routine requires "./Weno5_Routines.jl"
        # AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Xc0, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, dt, _dx, _dy, _dz, Ttop, Tbot )

        # New routine requires "./Weno5_Routines_v2.jl"
        AdvectWithWeno5_v2( Tc_ex, Tc_exxx, Xc0, Vx, Vy, Vz, dt, _dx, _dy, _dz, Ttop, Tbot )

        p1 = heatmap(xce, yce, (Tc_ex[:,:,2]'), c=cgrad(:hot, rev=true), aspect_ratio=:equal, xlim=(-1,1), ylims=(-1,1)) 
        display(plot(p1))

        _, I = findmax(Tc_ex[:,:,2])
        @printf("t = %f\n", time )
        @printf("x center position should be: %lf, current position: %lf, error = %2.2e\n", x0 + time*Vx[1], xce[I[1]], x0 + time*Vx[1] - xce[I[1]] )
        @printf("x center position should be: %lf, current position: %lf, error = %2.2e\n", y0 + time*Vy[1], yce[I[2]], y0 + time*Vy[1] - yce[I[2]] )
    end
end

TestAdvection()