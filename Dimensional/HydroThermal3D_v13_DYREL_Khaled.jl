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
using WriteVTK
plotlyjs()
# gr()

year = 365*3600*24

############################################## FUNCTIONS ##############################################

include("../tools/Macros.jl")  # Include macros - Cachemisère
include("../tools/Weno5_Routines_v2.jl")
include("./kernels_HT3D_dimensional.jl")
include("./kernels_HT3D_postprocessing.jl")

kTs(T)        = 3.1138 − 0.0023*(T)
kTf(T)        = -0.069 + 0.0012*(T)
Cf(T)         = 1000 + 7.5*(T)
Cs(T)         = 0.5915*(T) + 636.14
ρs(T)         = 2800*(1 - (0.000024*(T - 293)))
ρf_C(T_C,p)   = 1006+(7.424e-7*p)+(-0.3922*T_C)+(-4.441e-15*p^2)+(4.547e-9*p*T_C)+(-0.003774*T_C^2)+(1.451e-23*p^3)+(-1.793e-17*p^2*T_C)+(7.485e-12*p*T_C^2)+(2.955e-6*T_C^3)+(-1.463e-32*p^4)+(1.361e-26*p^3*T_C)+(4.018e-21*(p^2)*(T_C^2))+(-7.372e-15*p*T_C^3)+(5.698e-11*T_C^4)    
dρdP_C(T_C,p) = -7.372e-15 * T_C .^ 3 + 8.036e-21 * T_C .^ 2 .* p + 7.485e-12 * T_C .^ 2 + 4.083e-26 * T_C .* p .^ 2 - 3.586e-17 * T_C .* p + 4.547e-9 * T_C - 5.852e-32 * p .^ 3 + 4.353e-23 * p .^ 2 - 8.882e-15 * p + 7.424e-7
μf(T)         = 2.414e-5 * 10^(247.8/(T - 140.))
kF(y,δ)       = 5e-16*exp(y/δ)
ϕ(phase, ϕ0)  = phase == 2.0 ? 3*ϕ0 : ϕ0

a(x1, x2, y1, y2) = ( y1-y2 ) / ( x1-x2 )
b(x1, x2, y1, y2) = y1 - x1*(y1-y2)/(x1-x2)

@views function Topography( x, y_plateau, a2, b2 )
    # x intersect between function and y = 0 (west part)
    xW = -b2/a2
    # x intersect between function and y = y_plateau (east part)
    xE = (y_plateau - b2)/a2
    # y0 = 0.0
    if x<=xW 
        y0 = 0.0
    elseif x>= xE
        y0 = y_plateau
    else
        y0 = a2*x + b2
    end
    return y0
end

@views function SetInitialConditions_Khaled(phc, phv::Array{Float64,3}, geom, Tc_ex::Array{Float64,3}, Pc_ex, xce2::Array{Float64,3}, yce2::Array{Float64,3}, zce2::Array{Float64,3}, xv2::Array{Float64,3}, yv2::Array{Float64,3}, zv2::Array{Float64,3}, Tbot::Data.Number, Ttop::Data.Number, Pbot, Ptop, xmin::Data.Number, ymin::Data.Number, zmin::Data.Number, xmax::Data.Number, ymax::Data.Number, zmax::Data.Number, Ly::Data.Number, sticky_air, sc::scaling)
        xc2       = xce2[2:end-1,2:end-1,2:end-1]
        yc2       = yce2[2:end-1,2:end-1,2:end-1]
        zc2       = zce2[2:end-1,2:end-1,2:end-1]
        yv0       = zero(phv)
        yc0       = zero(phc)
        yce0      = zero(Tc_ex) 
        # Pluton position
        pluton = (x=50e3/sc.L, y=-17e3/sc.L, z=zmax/2, r=5e3/sc.L)   
        # Reference altitude
        yv0      .= Topography.(  xv2, geom.y_plateau, geom.surf.a1, geom.surf.b1 )
        yc0      .= Topography.(  xc2, geom.y_plateau, geom.surf.a1, geom.surf.b1 )
        yce0     .= Topography.( xce2, geom.y_plateau, geom.surf.a1, geom.surf.b1 )
        # Crust
        @. phv = 0.0
        @. phc = 0.0
        # Detachment 
        @. phv[ yv2 < (xv2*geom.fault.a1 + geom.fault.b1) && yv2 > (xv2*geom.fault.a2 + geom.fault.b2) && yv2 > (xv2*geom.fault.a3 + geom.fault.b3)  ] = 2.0
        @. phc[ yc2 < (xc2*geom.fault.a1 + geom.fault.b1) && yc2 > (xc2*geom.fault.a2 + geom.fault.b2) && yc2 > (xc2*geom.fault.a3 + geom.fault.b3)  ] = 2.0
        # Fault 1
        @. phv[ yv2 < (xv2*geom.fault1.a1 + geom.fault1.b1) && yv2 > (xv2*geom.fault1.a2 + geom.fault1.b2) && yv2 > (xv2*geom.fault1.a3 + geom.fault1.b3)  ] = 2.0
        @. phc[ yc2 < (xc2*geom.fault1.a1 + geom.fault1.b1) && yc2 > (xc2*geom.fault1.a2 + geom.fault1.b2) && yc2 > (xc2*geom.fault1.a3 + geom.fault1.b3)  ] = 2.0
        #Fault 2
        @. phv[ yv2 < (xv2*geom.fault2.a1 + geom.fault2.b1) && yv2 > (xv2*geom.fault2.a2 + geom.fault2.b2) && yv2 > (xv2*geom.fault2.a3 + geom.fault2.b3)  ] = 2.0
        @. phc[ yc2 < (xc2*geom.fault2.a1 + geom.fault2.b1) && yc2 > (xc2*geom.fault2.a2 + geom.fault2.b2) && yc2 > (xc2*geom.fault2.a3 + geom.fault2.b3)  ] = 2.0
        # Fault 3
        @. phv[ yv2 < (xv2*geom.fault3.a1 + geom.fault3.b1) && yv2 > (xv2*geom.fault3.a2 + geom.fault3.b2) && yv2 > (xv2*geom.fault3.a3 + geom.fault3.b3)  ] = 2.0
        @. phc[ yc2 < (xc2*geom.fault3.a1 + geom.fault3.b1) && yc2 > (xc2*geom.fault3.a2 + geom.fault3.b2) && yc2 > (xc2*geom.fault3.a3 + geom.fault3.b3)  ] = 2.0
        # Pluton
        @. phv[ ((xv2-pluton.x)^2 + (yv2-pluton.y)^2 + (zv2-pluton.z)^2) < pluton.r^2 ] = 3.0
        @. phc[ ((xc2-pluton.x)^2 + (yc2-pluton.y)^2 + (zc2-pluton.z)^2) < pluton.r^2 ] = 3.0
        # Air
        # if sticky_air
            @. phv[ yv2 > yv0 ] = 1.0
            @. phc[ yc2 > yc0 ] = 1.0
        # end
        # @. phv[ yv2>0 && (yv2 > (xv2*a2 + b2)) || yv2.>y_plateau ]                  = 1.0
        # @. k_ρf[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ] = Perm*k_ρf[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ]
        # Thermal field
        y_bot = -30e3/sc.L
        dTdy = (Ttop .- Tbot) ./ (yce0 .- y_bot)
        @. Tc_ex = Ttop + dTdy * (yce2 - yce0)
        @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:]
        @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:]
        @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1]
        @. Tc_ex[ yce2>0 && (yce2 > (xce2*geom.surf.a1 + geom.surf.b1)) || yce2.>geom.y_plateau] = Ttop
        # SET INITIAL THERMAL PERTUBATION
        # @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2 + (zce2-zmax/2)^2) < 0.01 ] += 0.1
        # Fluid pressure field
        y_bot = -30e3/sc.L
        dPdy = (Ptop .- Pbot) ./ (yce0 .- y_bot)
        @. Pc_ex = Ptop + dPdy * (yce2 - yce0)
        @. Pc_ex[1,:,:] =          Pc_ex[2,:,:]; @. Pc_ex[end,:,:] =          Pc_ex[end-1,:,:]
        @. Pc_ex[:,1,:] = 2*Pbot - Pc_ex[:,2,:]; @. Pc_ex[:,end,:] = 2*Ptop - Pc_ex[:,end-1,:]
        @. Pc_ex[:,:,1] =          Pc_ex[:,:,1]; @. Pc_ex[:,:,end] =          Pc_ex[:,:,end-1]
        @. Pc_ex[ yce2>0 && (yce2 > (xce2*geom.surf.a1 + geom.surf.b1)) || yce2.>geom.y_plateau] = Ptop
        return nothing
end

############################################## MAIN CODE ##############################################

@views function HydroThermal3D()

    @info "Starting HydroThermal3D!"

    # Visualise
    Hydro      = true
    Thermal    = true
    Advection  = true
    Vizu       = true
    Save       = true
    fact       = 1
    nt         = 100
    nout       = 10
    dt_fact    = 10
    sticky_air = false

    # Characteristic dimensions
    sc   = scaling()
    sc.T = 500.0
    sc.V = 1e-9
    sc.L = 700
    sc.η = 1e10
    scale_me!( sc )

    # Physics
    xmin     = -0.0/sc.L;  xmax = 120.0e3/sc.L;   Lx = xmax - xmin
    ymin     = -30e3/sc.L; ymax =    10e3/sc.L;   Ly = ymax - ymin
    zmin     = -20e3/sc.L; zmax =   20e3/sc.L;   Lz = zmax - zmin
    dT       = 600.0/sc.T
    Ttop     = 293.0/sc.T
    Tbot     = Ttop + dT
    dTdy     = (Ttop - Tbot)/Ly
    qyS      = 37e-3/(sc.W/sc.L^2)
    dP       = 294e6/sc.σ
    Ptop     = 1e5/sc.σ
    Pbot     = Ptop + dP
    ϕi       = 1e-2
    g        = -9.81/(sc.L/sc.t^2)
    Qt       = 3.4e-6/(sc.W/sc.L^3)  
    δ        = 3000. /sc.L
    k_fact   = 100.

    # Initial conditions 
    # Main detachment fault
    fault = (
        # top surface
        x1 = 75.8e3/sc.L, x2 = 31498.437753425103/sc.L,
        y1 = 3e3/sc.L,  y2 = -16897.187956165293/sc.L,
        # bottom surface
        x3 = 32000.0/sc.L, x4 = 77e3/sc.L, 
        y3 =-17800.0/sc.L, y4 = 3e3/sc.L,
        # left edge
        x5 = 32000.0/sc.L, x6 = 31498.437753425103/sc.L,
        y5 =-17800.0/sc.L, y6 = -16897.187956165293/sc.L,
    )
    # Secondary detachment fault 1
    fault1 = (
        # bottom surface
        x1 = 59.5e3/sc.L, x2 = 62.5e3/sc.L,  
        y1 = 0e3/sc.L,  y2 =-3e3/sc.L, 
        # top surface
        x3 =  61.8e3/sc.L, x4 = 58.8e3/sc.L,
        y3 =-3e3/sc.L, y4 = 0e3/sc.L, 
        # bottom
        x5 = 62.5e3/sc.L, x6 = 61.8e3/sc.L,
        y5 =-3e3/sc.L, y6 =-3e3/sc.L
        
    )
     # Secondary detachment fault 2
     fault2 = (
        # bottom surface
       x1 = 48.5e3/sc.L, x2 = 54.5e3/sc.L,  
       y1 = 0e3/sc.L,  y2 =-7e3/sc.L, 
       # top surface
       x3 =  53.8e3/sc.L, x4 = 47.8e3/sc.L,
       y3 =-7e3/sc.L, y4 = 0e3/sc.L, 
       # bottom
       x5 = 54.5e3/sc.L, x6 = 53.8e3/sc.L,
       y5 =-7e3/sc.L, y6 =-7e3/sc.L
   )
   # Secondary detachment fault 3
   fault3 = (
       # bottom surface
       x1 = 36.8e3/sc.L, x2 = 46.8e3/sc.L,  
       y1 = 0e3/sc.L,  y2 =-10.5e3/sc.L, 
       # top surface
       x3 =  46e3/sc.L, x4 = 36e3/sc.L,
       y3 = -10.5e3/sc.L, y4 = 0e3/sc.L, 
       # bottom
       x5 = 46.8e3/sc.L, x6 = 46e3/sc.L,
       y5 =-10.5e3/sc.L, y6 = -10.5e3/sc.L,
   )
    # Surface
    surf = (
        x1 = 32000.0/sc.L, x2 = 69397.54576150524/sc.L,
        y1 =-17800.0/sc.L, y2 = 0.0/sc.L,
    )
    
    geometry = (
        y_plateau = 0*3e3/sc.L,
        surf = (
            # Coefficient for surface slope
            a1 = a(surf.x1, surf.x2, surf.y1, surf.y2),
            b1 = b(surf.x1, surf.x2, surf.y1, surf.y2),
        ),
        fault = (
            # top
            a1 = a(fault.x1, fault.x2, fault.y1, fault.y2),
            b1 = b(fault.x1, fault.x2, fault.y1, fault.y2),
            # bottom
            a2 = a(fault.x3, fault.x4, fault.y3, fault.y4),
            b2 = b(fault.x3, fault.x4, fault.y3, fault.y4),
            # bottom
            a3 = a(fault.x5, fault.x6, fault.y5, fault.y6),
            b3 = b(fault.x5, fault.x6, fault.y5, fault.y6),
        ),
        fault1 = (
            # top   
            a1 = a(fault1.x1, fault1.x2, fault1.y1, fault1.y2),
            b1 = b(fault1.x1, fault1.x2, fault1.y1, fault1.y2),
            # bottom
            a2 = a(fault1.x3, fault1.x4, fault1.y3, fault1.y4),
            b2 = b(fault1.x3, fault1.x4, fault1.y3, fault1.y4),
            # bottom
            a3 = a(fault1.x5, fault1.x6, fault1.y5, fault1.y6),
            b3 = b(fault1.x5, fault1.x6, fault1.y5, fault1.y6),
        ),
        fault2 = (
            # top
            a1 = a(fault2.x1, fault2.x2, fault2.y1, fault2.y2),
            b1 = b(fault2.x1, fault2.x2, fault2.y1, fault2.y2),
            # bottom
            a2 = a(fault2.x3, fault2.x4, fault2.y3, fault2.y4),
            b2 = b(fault2.x3, fault2.x4, fault2.y3, fault2.y4),
            # bottom
            a3 = a(fault2.x5, fault2.x6, fault2.y5, fault2.y6),
            b3 = b(fault2.x5, fault2.x6, fault2.y5, fault2.y6),
        ),
        fault3 = (
            # top
            a1 = a(fault3.x1, fault3.x2, fault3.y1, fault3.y2),
            b1 = b(fault3.x1, fault3.x2, fault3.y1, fault3.y2),
            # bottom
            a2 = a(fault3.x3, fault3.x4, fault3.y3, fault3.y4),
            b2 = b(fault3.x3, fault3.x4, fault3.y3, fault3.y4),
            # bottom
            a3 = a(fault3.x5, fault3.x6, fault3.y5, fault3.y6),
            b3 = b(fault3.x5, fault3.x6, fault3.y5, fault3.y6),
        ),
    )

    @printf("Surface T = %03f, bottom T = %03f, qy = %03f\n", Ttop*sc.T, Tbot*sc.T, qyS*(sc.W/sc.L^2))

    # Numerics
    fact     = 16
    ncx      = fact*32-6
    ncy      = fact*8 -6
    # ncz      = fact*8 -6
    ncz      = 3#fact*32-6
    @info "ncx = $(ncx) ncy = $(ncy) ncz = $(ncz)"
    # Preprocessing
    if (USE_MPI) me, dims, nprocs, coords, comm = init_global_grid(ncx, ncy, ncz; dimx=2, dimy=2, dimz=2);             
    else         me, dims, nprocs, coords       = (0, [1,1,1], 1, [0,0,0]);
    end
    Nix      = USE_MPI ? nx_g() : ncx                                                
    Niy      = USE_MPI ? ny_g() : ncy                                                
    Niz      = USE_MPI ? nz_g() : ncz                                                
    dx, dy, dz = Lx/Nix, Ly/Niy, Lz/Niz                                            
    dt       = 1e1/sc.t  # min(dx,dy,dz)^2/6.1
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    _dt      = 1.0/dt
    # PT iteration parameters
    nitmax  = 1e4
    nitout  = 100
    tolT    = 1e-10  # Thermal solver
    tolP    = 1e-17  # Darcy solver
    @info "Go go go!!"

    # Initialisation
    phc      =  @ones(ncx+0,ncy+0,ncz+0) # Phase on centroids 
    phv      =  @ones(ncx+1,ncy+1,ncz+1) # Phase on vertices
    Xc0      = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    Xcit     = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    PC       =  @ones(ncx+0,ncy+0,ncz+0) # Common for P and T
    Fc       = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    Fcit     = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    Fc0      = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    ρC_ϕρ    = @zeros(ncx+0,ncy+0,ncz+0) # Common for P and T
    k_ρf     =  @ones(ncx+1,ncy+1,ncz+1) # Common for P and T
    dumc     =  @ones(ncx+0,ncy+0,ncz+0) # Common for P and T
    kx       = @zeros(ncx+1,ncy  ,ncz  ) # Common for P and T
    ky       = @zeros(ncx  ,ncy+1,ncz  ) # Common for P and T
    kz       = @zeros(ncx  ,ncy  ,ncz+1) # Common for P and T
    qx       = @zeros(ncx+1,ncy  ,ncz  ) # Common for P and T
    qy       = @zeros(ncx  ,ncy+1,ncz  ) # Common for P and T
    qz       = @zeros(ncx  ,ncy  ,ncz+1) # Common for P and T
    Vx       = @zeros(ncx+1,ncy  ,ncz  ) # Solution array 
    Vy       = @zeros(ncx  ,ncy+1,ncz  ) # Solution array 
    Vz       = @zeros(ncx  ,ncy  ,ncz+1) # Solution array 
    Tc_ex    = @zeros(ncx+2,ncy+2,ncz+2) # Solution array
    Pc_ex    = @zeros(ncx+2,ncy+2,ncz+2) # Solution array
    # Advection stuff
    Tc_exxx  = @zeros(ncx+6,ncy+6,ncz+6)
    @info "Memory was allocated!"

    # Pre-processing
    if USE_MPI
        xc  = [x_g(ix,dx,Fc)+dx/2 for ix=1:ncx]
        yc  = [y_g(iy,dy,Fc)+dy/2 for iy=1:ncy]
        zc  = [z_g(iz,dz,Fc)+dz/2 for iz=1:ncz]
        xce = [x_g(ix,dx,Fc)-dx/2 for ix=1:ncx+2]
        yce = [y_g(iy,dy,Fc)-dy/2 for iy=1:ncy+2]
        zce = [z_g(iz,dz,Fc)-dz/2 for iz=1:ncz+2]
        xv  = [x_g(ix,dx,Fc) for ix=1:ncx]
        yv  = [y_g(iy,dy,Fc) for iy=1:ncy]
        zv  = [z_g(iz,dz,Fc) for iz=1:ncz]
    else
        xc  = LinRange(xmin+dx/2, xmax-dx/2, ncx)
        yc  = LinRange(ymin+dy/2, ymax-dy/2, ncy)
        zc  = LinRange(xmin+dz/2, zmax-dz/2, ncz)
        xce = LinRange(xmin-dx/2, xmax+dx/2, ncx+2)
        yce = LinRange(ymin-dy/2, ymax+dy/2, ncy+2)
        zce = LinRange(xmin-dz/2, zmax+dz/2, ncz+2)
        xv  = LinRange(xmin, xmax, ncx+1)
        yv  = LinRange(ymin, ymax, ncy+1)
        zv  = LinRange(xmin, zmax, ncz+1)
    end
    (xce2,yce2,zce2) = ([x for x=xce,y=yce,z=zce], [y for x=xce,y=yce,z=zce], [z for x=xce,y=yce,z=zce]) #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
    (xv2,yv2,zv2)    = ([x for x=xv,y=yv,z=zv],    [y for x=xv,y=yv,z=zv],    [z for x=xv,y=yv,z=zv]   )
    @printf("Grid was set up!\n")

    Tc_ex = Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE CPU:
    Pc_ex = Array(Pc_ex)
    phv   = Array(phv)   # Ensure it is temporarily a CPU array
    phc   = Array(phc)   # Ensure it is temporarily a CPU array
    SetInitialConditions_Khaled(phc, phv, geometry, Tc_ex, Pc_ex, xce2, yce2, zce2, xv2, yv2, zv2, Tbot, Ttop, Pbot, Ptop, xmin, ymin, zmin, xmax, ymax, zmax, Ly, sticky_air, sc)
    Tc_ex = Data.Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE GPU
    Pc_ex = Data.Array(Pc_ex) # MAKE SURE ACTIVITY IS IN THE GPU
    phv   = Data.Array(phv)
    phc   = Data.Array(phc)

    @printf("min(Tc_ex) = %11.4e - max(Tc_ex) = %11.4e\n", minimum_g(Tc_ex)*sc.T, maximum_g(Tc_ex)*sc.T )
    @printf("min(Pc_ex) = %11.4e - max(Pc_ex) = %11.4e\n", minimum_g(Pc_ex)*sc.σ, maximum_g(Pc_ex)*sc.σ )
    @printf("min(Vy)    = %11.4e - max(Vy)    = %11.4e\n", minimum_g(Vy)*sc.V,    maximum_g(Vy)*sc.V )

    it1=0; time=0; transient = 1.0

    ## Action
    @time for it = it1:nt

        if it==0 
            @printf("\n/******************* Initialisation step *******************\n")
            transient = 0.0
        else
            @printf("\n/******************* Time step %05d *******************\n", it)
            transient = 1.0
        end

        if Thermal
            @printf(">>>> Thermal solver\n");
            @parallel ComputeThermalConductivity( k_ρf, Tc_ex, phv, ϕi, sc.kt, sc.T )
            @parallel SmoothConductivityV2C( dumc,  k_ρf )
            @parallel SmoothConductivityC2V( k_ρf, dumc )
            @parallel InitConductivity!(kx, ky, kz, k_ρf)
            @parallel ComputeρCeffective!(ρC_ϕρ, Tc_ex, Pc_ex, phc, ϕi, sc.T, sc.σ, sc.C, sc.ρ)

            @parallel GershgorinPoisson!( Xc0, PC, ρC_ϕρ, kx, ky, kz, transient, dt, dx, dy, dz )
            λmax     = maximum_g(Xc0)
            λmin     = λmax / 500
            CFL_T    = 0.99
            cfact    = 0.9
            Δτ       = 2.0./sqrt.(λmax)*CFL_T
            c        = 2.0*sqrt(λmin)*cfact
            h1, h2   = (2-c*Δτ)/(2+c*Δτ), 2*Δτ/(2+c*Δτ)
            @show λmin, λmax
    
            @parallel ResetA!(Fc, Fc0)
            @parallel InitThermal!(Xc0, Tc_ex)
            nF_abs, nF_rel, nF_ini = 0., 0., 0.

            # Iteration loop
            for iter = 1:nitmax

                check = mod(iter,nitout) == 0 || iter<=2
                if check @parallel SwapDYREL!(Xcit, Fcit, Tc_ex, Fc) end
                @parallel SetTemperatureBCs!(Tc_ex, phc, qyS, _dy, 1.0/sc.kt, Ttop,  geometry.y_plateau, geometry.surf.a1, geometry.surf.b1, dTdy, dx, dy, sticky_air)
                @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Tc_ex, _dx, _dy, _dz)
                @parallel ResidualTemperatureLinearised!(Fc, Tc_ex, Xc0, ρC_ϕρ, phc, PC, qx, qy, qz, Qt, transient, _dt, _dx, _dy, _dz)
                @parallel DYRELUpdate!(Fc0, Tc_ex, Fc, h1, h2, Δτ)
                
                if (USE_MPI) update_halo!(Tc_ex); end
                if check
                    @parallel Multiply( dumc, Fc, PC )
                    nF_abs = mean_g(abs.(dumc))/sqrt(ncx*ncy*ncz) * (sc.ρ*sc.C*sc.T/sc.t)
                    if iter==1 nF_ini = nF_abs end
                    nF_rel = nF_abs/nF_ini
                    if (me==0) @printf("PT iter. #%05d - || Fc_abs || = %2.2e - || Fc_rel || = %2.2e\n", iter, nF_abs, nF_rel) end
                    if (me==0) if isnan(nF_abs) error("Nan T...")      end end
                    if (me==0) if nF_rel>100    error("Diverged T...") end end

                    if iter>1
                        @parallel RayleighQuotientNumerator!( dumc, Fc0, Fcit, Fc, Δτ )
                        λmin   = abs(sum_g(dumc)) 
                        @parallel RayleighQuotientDenominator!( dumc, Fc0, Δτ )
                        λmin  /= abs(sum_g(dumc))
                        c      = 2.0*sqrt(λmin)*cfact
                        h1, h2 = (2-c*Δτ)/(2+c*Δτ), 2*Δτ/(2+c*Δτ)
                    end

                    if (nF_abs<tolT || nF_rel<tolT)  break end
                end
            end
        end

        if Hydro
            @printf(">>>> Darcy solver\n");
            @parallel ComputeHydroConductivity(k_ρf, Tc_ex, Pc_ex, phv, ymin, dy, k_fact, δ, sc.σ, sc.T, sc.L, sc.t, sc.η, 1)
            @parallel SmoothConductivityV2C( dumc, k_ρf )
            @parallel SmoothConductivityC2V( k_ρf, dumc )
            @parallel InitConductivity!(kx, ky, kz, k_ρf)

            @parallel ϕdρdP!(ρC_ϕρ, Tc_ex, Pc_ex, phc, ϕi, sc.ρ, sc.T, sc.σ )   
            @parallel GershgorinPoisson!( Xc0, PC, ρC_ϕρ, kx, ky, kz, transient, dt, dx, dy, dz )
            λmax     = maximum_g(Xc0)
            λmin     = λmax / 500
            CFL_P    = 1.0
            cfact    = 0.9
            Δτ       = 2.0./sqrt.(λmax)*CFL_P
            c        = 2.0*sqrt(λmin)*cfact
            h1, h2   = (2-c*Δτ)/(2+c*Δτ), 2*Δτ/(2+c*Δτ)
            @show λmin, λmax

            @parallel ResetA!(Fc, Fc0)
            @parallel ϕρf(Xc0, Tc_ex, Pc_ex, phc, ϕi, sc.ρ, sc.T, sc.σ)
            nF_abs, nF_rel, nF_ini = 0., 0., 0.
            @parallel ComputeFluidDensity(k_ρf, Tc_ex, Pc_ex, phv, sc.σ, sc.T, sc.ρ)

            # Iteration loop
            for iter = 1:nitmax

                check = mod(iter,nitout) == 0 || iter<=2
                if check @parallel SwapDYREL!(Xcit, Fcit, Pc_ex, Fc) end
                @parallel ϕρf(ρC_ϕρ, Tc_ex, Pc_ex, phc, ϕi, sc.ρ, sc.T, sc.σ)
                @parallel SetPressureBCs!(Pc_ex, phc, Pbot, Ptop, geometry.y_plateau, geometry.surf.a1, geometry.surf.b1, 1000/sc.ρ, g, dx, dy, sticky_air)
                @parallel ComputeDarcyFlux!(qx, qy, qz, k_ρf, kx, ky, kz, Pc_ex, g, _dx, _dy, _dz)
                @parallel ResidualFluidPressure!(Fc, phc, ρC_ϕρ, Xc0, PC, qx, qy, qz, transient, _dt, _dx, _dy, _dz)
                @parallel DYRELUpdate!(Fc0, Pc_ex, Fc, h1, h2, Δτ)

                if (USE_MPI) update_halo!(Pc_ex); end
                if check
                    @parallel Multiply( dumc, Fc, PC )
                    nF_abs = mean_g(abs.(dumc))/sqrt(ncx*ncy*ncz) * (sc.ρ/sc.t)
                    if iter==1 nF_ini = nF_abs end
                    nF_rel = nF_abs/nF_ini
                    if (me==0) @printf("PT iter. #%05d - || Fc_abs || = %2.2e - || Fc_rel || = %2.2e\n", iter, nF_abs, nF_rel) end
                    if (me==0) if isnan(nF_abs) error("Nan P...")      end end
                    if (me==0) if nF_rel>100    error("Diverged P...") end end

                    if iter>1
                        @parallel RayleighQuotientNumerator!( dumc, Fc0, Fcit, Fc, Δτ )
                        λmin   = abs(sum_g(dumc)) 
                        @parallel RayleighQuotientDenominator!( dumc, Fc0, Δτ )
                        λmin  /= abs(sum_g(dumc))
                        c      = 2.0*sqrt(λmin)*cfact
                        h1, h2 = (2-c*Δτ)/(2+c*Δτ), 2*Δτ/(2+c*Δτ)
                    end

                    if (nF_abs<tolP || nF_rel<tolP)  break end
                end
            end
        end

        # Compute velocity - here uses k_ρf = k/μ (does not include ϕ)
        @parallel ComputeHydroConductivity(k_ρf, Tc_ex, Pc_ex, phv, ymin, dy, k_fact, δ, sc.σ, sc.T, sc.L, sc.t, sc.η, 2)
        @parallel SmoothConductivityV2C( dumc, k_ρf )
        @parallel SmoothConductivityC2V( k_ρf, dumc )
        @parallel InitConductivity!(kx, ky, kz, k_ρf)
        @parallel ComputeFluidDensity(k_ρf, Tc_ex, Pc_ex, phv, sc.σ, sc.T, sc.ρ)
        @parallel ComputeDarcyFlux!(Vx, Vy, Vz, k_ρf, kx, ky, kz, Pc_ex, g, _dx, _dy, _dz)

        if it>0
            time  = time + dt;
            @printf("\n-> it=%d, time=%.1e, dt=%.1e, \n", it, time, dt);

            #---------------------------------------------------------------------
            if Advection
                AdvectWithWeno5_v2( Tc_ex, Tc_exxx, Xc0, Vx, Vy, Vz, dt, _dx, _dy, _dz, Ttop, Tbot )

                # Set dt for next step
                dt  = dt_fact*1.0/6.1*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
                _dt = 1.0/dt
                @printf("Time step = %2.2e s\n", dt*sc.t)
            end
        end

        @printf("min(Tc_ex) = %11.4e - max(Tc_ex) = %11.4e\n", minimum_g(Tc_ex)*sc.T, maximum_g(Tc_ex)*sc.T )
        @printf("min(Pc_ex) = %11.4e - max(Pc_ex) = %11.4e\n", minimum_g(Pc_ex)*sc.σ, maximum_g(Pc_ex)*sc.σ )
        @printf("min(Vy)    = %11.4e - max(Vy)    = %11.4e\n", minimum_g(Vy)*sc.V,    maximum_g(Vy)*sc.V )

        #---------------------------------------------------------------------
        if (Vizu && mod(it, nout) == 0)
            tMa = @sprintf("%03f", time*sc.t/1e6/year)
            y_topo = Topography.( xce, geometry.y_plateau, geometry.surf.a1, geometry.surf.b1 )
            p1 = heatmap(xce*sc.L/1e3, yce*sc.L/1e3, (Tc_ex[:,:,2]'.*sc.T.-273.15), c=cgrad(:hot, rev=true), aspect_ratio=1, xlim=(0,120), ylim=(-30,5))  #, clims=(0, 700)
            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, phv[:,:,2]')
            p2 = heatmap(xce*sc.L/1e3, yce*sc.L/1e3, (Pc_ex[:,:,2]'.*sc.σ./1e6), c=:jet1, aspect_ratio=1, xlim=(0,120), ylim=(-30,5)) 
            p3 = heatmap(xc *sc.L/1e3, yv *sc.L/1e3, (Vy[:,:,2]'.*sc.V*100*year), c=:jet1, aspect_ratio=1, clims=(-27, 23), xlim=(0,120), ylim=(-30,5)) #title="Vy [cm/y]"*string(" @ t = ", tMa, " My" ) 
            
            x = Tc_ex[:,55,2]
            p3 = plot(xce*sc.L/1e3, x.*sc.T.-273.15)
            
            p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, phv[:,:,2]', c=:jet1, aspect_ratio=1, xlim=(0,120), ylim=(-30,5)) 

            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (k_ρf[:,:,2]'.*sc.kt), c=:jet1, aspect_ratio=1) 
            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (k_ρf[:,:,2]'.*sc.t), c=:jet1, aspect_ratio=1) 


            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, log10.(k_ρf[:,:,2]'.*sc.t), c=:jet1, aspect_ratio=1) 
            # X = Tc_ex[2:end-1,2:end-1,2:end-1]
            #  heatmap(xc, yc, transpose(X[:,:,Int(ceil(ncz/2))]),c=:viridis,aspect_ratio=1) 
            #heatmap(xc*sc.L/1e3, yc*sc.L/1e3, (Fc[:,:,1]'.*(sc.ρ/sc.t)), c=:jet1, aspect_ratio=1) 
            #  heatmap(xv*sc.L/1e3, yv*sc.L/1e3, log10.(kf2[:,:,2]'.*sc.kf), c=:jet1, aspect_ratio=1) 
            #   contourf(xc,yc,transpose(Ty[:,:,Int(ceil(ncz/2))])) ) # accede au sublot 111
            #quiver(x,y,(f,f))
            # p1 = plot!(xce*sc.L/1e3, y_topo*sc.L/1e3, c=:white)
            _, izero =  findmin(abs.(yv))
            # p1 = plot!()
            # p2 = plot!(xlim=(0,120), ylim=(-30,5))
            # p3 = plot!(xlim=(0,120), ylim=(-30,5))
            p6 = plot(xc*sc.L/1e3, Vy[:,izero,2].*sc.V*100*year, label=:none)
            display(plot(p1, p2, p3, layout=(3,1)))
            @printf("Imaged sliced at z index %d over ncx = %d, ncy = %d, ncz = %d --- time is %02f Ma\n", Int(ceil(ncz/2)), ncx, ncy, ncz, time*sc.t/1e6/year)
            #  heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1) 
        end
        #---------------------------------------------------------------------

        if ( Save && mod(it, nout) == 0 )
            filename = @sprintf("./HT3DOutput%05d", it)
            vtkfile  = vtk_grid(filename, Array(xc), Array(yc), Array(zc))
            @parallel Phase!( dumc, phc )
            vtkfile["Phase"]       = Array(dumc)
            @parallel Pressure!( dumc, Pc_ex, phc, sc.σ )
            vtkfile["Pressure"]    = Array(dumc)
            @parallel Temperature!( dumc, Tc_ex, phc, sc.T )
            vtkfile["Temperature"] = Array(dumc)
            @parallel Velocity!( Fc0, Fc, Fcit, Vx, Vy, Vz, phc, sc.V )
            vtkfile["Velocity"]    = (Array(Fc0), Array(Fc), Array(Fcit))
            @parallel EffectiveThermalConductivity!( dumc, Tc_ex, ϕi, phc, sc.T )
            vtkfile["kThermal"]    = Array(dumc)
            @parallel Permeability!( dumc, k_fact, ymin, dy, phc, δ, sc.L )
            vtkfile["Perm"]        = Array(dumc)
            @parallel FluidDensity!( dumc, Tc_ex, Pc_ex, phc, sc.T, sc.σ )
            vtkfile["Density"]     = Array(dumc)
            @parallel Viscosity!( dumc, Tc_ex, phc, sc.T )
            vtkfile["Viscosity"]   = Array(dumc)
            @parallel Peclet!( dumc, Tc_ex, Pc_ex, Vx, Vy, Vz, phc, ϕi, sc.T, sc.σ, sc.V, sc.L )
            vtkfile["Peclet"]   = Array(dumc)
            outfiles = vtk_save(vtkfile)
        end
        #---------------------------------------------------------------------

end#it
#
# if me == 0
#     # println("time_s=$time_s T_eff=$T_eff niter=$niter iterMin=$iterMin iterMax=$iterMax");
#     # println("nprocs $nprocs dims $(dims[1]) $(dims[2]) $(dims[3]) fdims 1 1 1 nxyz $ncx $ncy $ncz nt $(iterMin-warmup) nb_it 1 PRECIS $(sizeof(Data.Number)) time_s $time_s block $(cuthreads[1]) $(cuthreads[2]) $(cuthreads[3]) grid $(cublocks[1]) $(cublocks[2]) $(cublocks[3])\n");
#     gif(anim, "Diffusion_fps15_1.gif", fps=15);
# end
# if (USE_MPI) finalize_global_grid(); end

end 

HydroThermal3D()