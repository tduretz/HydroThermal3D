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
plotlyjs()

year = 365*3600*24

include("./tools/Macros.jl")  # Include macros - Cachemisère
include("./tools/Weno5_Routines.jl")
include("./kernels_HT3D_dimensional.jl")

kTs(T)    = 3.1138 − 0.0023*(T)
kTf(T)    = -0.069 + 0.0012*(T)
Cf(T)     = 1000 + 7.5*(T)
Cs(T)     = 0.5915*(T) + 636.14
ρs(T)     = 2800*(1 - (0.000024*(T - 293)))
ρf(T,p)   = 1006+(7.424e-7*p)+(-0.3922*(T-273.15))+(-4.441e-15*p^2)+(4.547e-9*p*(T-273.15))+(-0.003774*(T-273.15)^2)+(1.451e-23*p^3)+(-1.793e-17*p^2*(T-273.15))+(7.485e-12*p*(T-273.15)^2)+(2.955e-6*(T-273.15)^3)+(-1.463e-32*p^4)+(1.361e-26*p^3*(T-273.15))+(4.018e-21*(p^2)*((T-273.15)^2))+(-7.372e-15*p*(T-273.15)^3)+(5.698e-11*(T-273.15)^4)    
ρf_C(T_C,p)   = 1006+(7.424e-7*p)+(-0.3922*T_C)+(-4.441e-15*p^2)+(4.547e-9*p*T_C)+(-0.003774*T_C^2)+(1.451e-23*p^3)+(-1.793e-17*p^2*T_C)+(7.485e-12*p*T_C^2)+(2.955e-6*T_C^3)+(-1.463e-32*p^4)+(1.361e-26*p^3*T_C)+(4.018e-21*(p^2)*(T_C^2))+(-7.372e-15*p*T_C^3)+(5.698e-11*T_C^4)    

μf(T)     = 2.414e-5 * 10^(247.8/(T - 140.))
kF(y,δ)   = 5e-16*exp(y/δ)
function ϕ(phase, ϕ0)
    if phase == 2.0
        return ϕ = 3*ϕ0
    else
        return ϕ   = ϕ0
    end
end

############################################## MAIN CODE ##############################################

function Topography( x, y_plateau, a2, b2 )
    # x intersect beween function and y = 0 (west part)
    xW = -b2/a2
    # x intersect beween function and y = y_plateau (east part)
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

@parallel_indices (i,j,k) function UpdateThermalConductivity( ktv, Tce, phv, ϕ0, scale_kt, scale_T )
    if i<=size(ktv, 1) && j<=size(ktv, 2) && k<=size(ktv, 3) 
        # Interpolate from extended centroids to vertices
        Tv  = 1.0/8.0*(Tce[i+1,j+1,k+1] + Tce[i+1,j,k+1] + Tce[i,j+1,k+1] + Tce[i,j,k+1])
        Tv += 1.0/8.0*(Tce[i+1,j+1,k+0] + Tce[i+1,j,k+0] + Tce[i,j+1,k+0] + Tce[i,j,k+0])
        Tv *= scale_T
        # Effective conductivity
        ktv[i,j,k] = ((1-ϕ(phv[i,j,k],ϕ0))*kTs(Tv) + ϕ(phv[i,j,k],ϕ0)*kTf(Tv))/scale_kt
        # Air
        # if phv[i,j,k] == 1.0
        #     ktv[i,j,k] = 200.0/scale_kt
        # end
    end
    return nothing
end

@parallel_indices (i,j,k) function UpdateHydroConductivity(kv, kfv, ρfv, Tce, Pce, yv2, phv, scale_σ, scale_T, scale_L, scale_t, scale_ρ, scale_η)
    if i<=size(kfv, 1) && j<=size(kfv, 2) && k<=size(kfv, 3) 
        # Interpolate from extended centroids to vertices
        Tv  = 1.0/8.0*(Tce[i+1,j+1,k+1] + Tce[i+1,j,k+1] + Tce[i,j+1,k+1] + Tce[i,j,k+1])
        Tv += 1.0/8.0*(Tce[i+1,j+1,k+0] + Tce[i+1,j,k+0] + Tce[i,j+1,k+0] + Tce[i,j,k+0])
        Tv *= scale_T
        Pv  = 1.0/8.0*(Pce[i+1,j+1,k+1] + Pce[i+1,j,k+1] + Pce[i,j+1,k+1] + Pce[i,j,k+1])
        Pv += 1.0/8.0*(Pce[i+1,j+1,k+0] + Pce[i+1,j,k+0] + Pce[i,j+1,k+0] + Pce[i,j,k+0])
        Pv *= scale_σ
        δ   = 3000. /scale_L
        ρk_μ = ρf(Tv, Pv) * kF(yv2[i,j,k], δ) /  μf(Tv)
        # if i==2 && k==2
        #     @show μf(Tv)*1e3
        #     # @show Tv, Pv/1e6
        # end
        kv[i,j,k]  = (kF(yv2[i,j,k], δ) /  μf(Tv)) / (scale_L^2/scale_η)
        # if phv[i,j,k] == 1.0
        #     kv[i,j,k]  = 0.
        # end
        ρfv[i,j,k] = ρf(Tv, Pv) / scale_ρ
        kfv[i,j,k] = ρk_μ / scale_t
        # Fault
        if phv[i,j,k] == 2.0
            kfv[i,j,k] *= 100
        end
        # Air
        if phv[i,j,k] == 1.0
            ρfv[i,j,k]  = 0.000  / scale_ρ
            # kfv[i,j,k] *= 100 # 1e-9 / scale_t
        end
    end
    return nothing
end

@parallel function SmoothConductivityV2C( ktc, ktv )
    @all(ktc) = @av(ktv)
    return nothing
end

@parallel function SmoothConductivityC2V( ktv, ktc )
        @inn(ktv) = @av(ktc)
    return nothing
end

@views function SetInitialConditions_Khaled(phc, phv::Array{Float64,3}, geom, Tc_ex::Array{Float64,3}, Pc_ex, xce2::Array{Float64,3}, yce2::Array{Float64,3}, zce2::Array{Float64,3}, xv2::Array{Float64,3}, yv2::Array{Float64,3}, zv2::Array{Float64,3}, Tbot::Data.Number, Ttop::Data.Number, Pbot, Ptop, xmin::Data.Number, ymin::Data.Number, zmin::Data.Number, xmax::Data.Number, ymax::Data.Number, zmax::Data.Number, Ly::Data.Number, sticky_air, sc::scaling)
        xc2       = xce2[2:end-1,2:end-1,2:end-1]
        yc2       = yce2[2:end-1,2:end-1,2:end-1]
        zc2       = zce2[2:end-1,2:end-1,2:end-1]
        yv0       = zero(phv)
        yc0       = zero(phc)
        yce0      = zero(Tc_ex)        
        # Reference altitude
        yv0      .= Topography.(  xv2, geom.y_plateau, geom.a2, geom.b2 )
        yc0      .= Topography.(  xc2, geom.y_plateau, geom.a2, geom.b2 )
        yce0     .= Topography.( xce2, geom.y_plateau, geom.a2, geom.b2 )
        # Crust
        @. phv = 0.0
        @. phc = 0.0
        # Fault
        @. phv[ yv2 < (xv2*geom.a1 + geom.b1) && yv2 > (xv2*geom.a2 + geom.b2) && yv2 > (xv2*geom.a3 + geom.b3)  ] = 2.0
        @. phc[ yc2 < (xc2*geom.a1 + geom.b1) && yc2 > (xc2*geom.a2 + geom.b2) && yc2 > (xc2*geom.a3 + geom.b3)  ] = 2.0
        # Pluton
        r_pluton = 3e3/sc.L
        @. phv[ ((xv2-xmax/2)^2 + (yv2-ymin/3)^2 + (zv2-zmax/2)^2) < r_pluton^2 ] = 3.0
        @. phc[ ((xc2-xmax/2)^2 + (yc2-ymin/3)^2 + (zc2-zmax/2)^2) < r_pluton^2 ] = 3.0
        # Air
        # if sticky_air
            @. phv[ yv2 > yv0 ] = 1.0
            @. phc[ yc2 > yc0 ] = 1.0
        # end
        # @. phv[ yv2>0 && (yv2 > (xv2*a2 + b2)) || yv2.>y_plateau ]                  = 1.0
        # @. kfv[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ] = Perm*kfv[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ]
        # Thermal field
        y_bot = -30e3/sc.L
        dTdy = (Ttop .- Tbot) ./ (yce0 .- y_bot)
        @. Tc_ex = Ttop + dTdy * (yce2 - yce0)
        @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:]
        @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:]
        @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1]
        @. Tc_ex[ yce2>0 && (yce2 > (xce2*geom.a2 + geom.b2)) || yce2.>geom.y_plateau] = Ttop
        # SET INITIAL THERMAL PERTUBATION
        # @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2 + (zce2-zmax/2)^2) < 0.01 ] += 0.1
        # Fluid pressure field
        y_bot = -30e3/sc.L
        dPdy = (Ptop .- Pbot) ./ (yce0 .- y_bot)
        @. Pc_ex = Ptop + dPdy * (yce2 - yce0)
        @. Pc_ex[1,:,:] =          Pc_ex[2,:,:]; @. Pc_ex[end,:,:] =          Pc_ex[end-1,:,:]
        @. Pc_ex[:,1,:] = 2*Pbot - Pc_ex[:,2,:]; @. Pc_ex[:,end,:] = 2*Ptop - Pc_ex[:,end-1,:]
        @. Pc_ex[:,:,1] =          Pc_ex[:,:,1]; @. Pc_ex[:,:,end] =          Pc_ex[:,:,end-1]
        @. Pc_ex[ yce2>0 && (yce2 > (xce2*geom.a2 + geom.b2)) || yce2.>geom.y_plateau] = Ptop
        return nothing
end

############################################## MAIN CODE ##############################################

@views function HydroThermal3D()

    @info "Starting HydroThermal3D!"

    # Visualise
    Advection  = 1
    Vizu       = 1
    Save       = 0
    fact       = 1
    nt         = 0
    nout       = 1
    dt_fact    = 10
    sticky_air = false

    # Characteristic dimensions
    sc   = scaling()
    sc.T = 500.0
    sc.V = 1e-9
    sc.L = 100e3
    sc.η = 1e10
    scale_me!( sc )

    # Physics
    xmin     = -0.0/sc.L;  xmax = 120.0e3/sc.L; 
    ymin     = -30e3/sc.L; ymax =  0e3/sc.L;    
    zmin     = -0.00/sc.L; zmax = 1.0e3/sc.L;   
    if sticky_air
        ymax = 10e3/sc.L
    end

    ymax = 10e3/sc.L
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

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

    # Initial conditions: Draw Fault
    # top
    x1 = 68466.18089117216/sc.L; x2 = 31498.437753425103/sc.L
    y1 = 0.0/sc.L;               y2 = -16897.187956165293/sc.L
    # bottom
    x3 = 32000.0/sc.L; x4 = 69397.54576150524/sc.L
    y3 =-17800.0/sc.L; y4 = 0.0/sc.L
    # bottom
    x5 = 32000.0/sc.L; x6 = 31498.437753425103/sc.L
    y5 =-17800.0/sc.L; y6 = -16897.187956165293/sc.L
    
    geometry = (
        y_plateau = 1*3e3/sc.L,
        # top
        a1 = ( y1-y2 ) / ( x1-x2 ),
        b1 = y1 - x1*(y1-y2)/(x1-x2),
        # bottom
        a2 = ( y3-y4 ) / ( x3-x4 ),
        b2 = y3 - x3*(y3-y4)/(x3-x4),
        # bottom
        a3 = ( y5-y6 ) / ( x5-x6 ),
        b3 = y5 - x5*(y5-y6)/(x5-x6),
    )

    @printf("Surface T = %03f, bottom T = %03f, qy = %03f\n", Ttop*sc.T, Tbot*sc.T, qyS*(sc.W/sc.L^2))

    # Numerics
    fact     = 16
    ncx      = fact*32-6
    ncy      = fact*8 -6
    ncz      = 3#fact*32-6
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
    nitmax  = 5e3
    nitout  = 100
    # Thermal solver
    tolT    = 1e-9
    tetT    = 0.1
    Tdamp   = 0.025
    dampT   = 1*(1-Tdamp/min(ncx,ncy,ncz))
    dtauT   = (tetT*min(dx,dy,dz)^2/4.1)/sc.t * 1e4
    dtauT   = (tetT*min(dx,dy,dz)^2/4.1)/sc.t * 200

    # Darcy solver
    tolP     = 1e-14
    tetP     = 1/4/3
    dtauP    = tetP/6.1*min(dx,dy,dz)^2/sc.t * 7e36
    Pdamp    = 0.1
    dampP    = 1*(1-Pdamp/min(ncx,ncy,ncz)) 
    @info "Go go go!!"

    # Initialisation
    Tc0      = @zeros(ncx+0,ncy+0,ncz+0)
    ϕρf0c    = @zeros(ncx+0,ncy+0,ncz+0)
    ϕρfc     = @zeros(ncx+0,ncy+0,ncz+0)
    ρC_eff   = @zeros(ncx+0,ncy+0,ncz+0)
    ρfv      = @ones(ncx+1,ncy+1,ncz+1)
    v1       = @zeros(ncx+0,ncy+0,ncz+0)
    v2       = @zeros(ncx+0,ncy+0,ncz+0)
    v3       = @zeros(ncx+0,ncy+0,ncz+0)
    v4       = @zeros(ncx+0,ncy+0,ncz+0)
    v5       = @zeros(ncx+0,ncy+0,ncz+0)
    dTdxp    = @zeros(ncx+0,ncy+0,ncz+0)
    dTdxm    = @zeros(ncx+0,ncy+0,ncz+0)
    Tc       = @zeros(ncx+0,ncy+0,ncz+0)
    Tc_ex    = @zeros(ncx+2,ncy+2,ncz+2)
    Tc_exxx  = @zeros(ncx+6,ncy+6,ncz+6)
    Pc_ex    = @zeros(ncx+2,ncy+2,ncz+2)
    kfv      =  @ones(ncx+1,ncy+1,ncz+1)
    kv       =  @ones(ncx+1,ncy+1,ncz+1) # to delete
    phc      =  @ones(ncx+0,ncy+0,ncz+0)
    phv      =  @ones(ncx+1,ncy+1,ncz+1)
    ktv      =  @ones(ncx+1,ncy+1,ncz+1)
    kc       =  @ones(ncx+0,ncy+0,ncz+0)
    kx       = @zeros(ncx+1,ncy  ,ncz  )
    ky       = @zeros(ncx  ,ncy+1,ncz  )
    kz       = @zeros(ncx  ,ncy  ,ncz+1)
    qx       = @zeros(ncx+1,ncy  ,ncz  )
    qy       = @zeros(ncx  ,ncy+1,ncz  )
    qz       = @zeros(ncx  ,ncy  ,ncz+1)
    Vx       = @zeros(ncx+1,ncy  ,ncz  )
    Vy       = @zeros(ncx  ,ncy+1,ncz  )
    Vz       = @zeros(ncx  ,ncy  ,ncz+1)
    Ft       = @zeros(ncx  ,ncy  ,ncz  )
    Ft0      = @zeros(ncx  ,ncy  ,ncz  )
    Vxm      = @zeros(ncx+0,ncy+0,ncz+0)
    Vxp      = @zeros(ncx+0,ncy+0,ncz+0)
    Vym      = @zeros(ncx+0,ncy+0,ncz+0)
    Vyp      = @zeros(ncx+0,ncy+0,ncz+0)
    Vzm      = @zeros(ncx+0,ncy+0,ncz+0)
    Vzp      = @zeros(ncx+0,ncy+0,ncz+0)
    @info "Memory was allocated!"

    # Pre-processing
    if USE_MPI
        xc  = [x_g(ix,dx,Ft)+dx/2 for ix=1:ncx];
        yc  = [y_g(iy,dy,Ft)+dy/2 for iy=1:ncy];
        zc  = [z_g(iz,dz,Ft)+dz/2 for iz=1:ncz];
        xce = [x_g(ix,dx,Ft)-dx/2 for ix=1:ncx+2]; # ACHTUNG
        yce = [y_g(iy,dy,Ft)-dy/2 for iy=1:ncy+2]; # ACHTUNG
        zce = [z_g(iz,dz,Ft)-dz/2 for iz=1:ncz+2]; # ACHTUNG
        xv  = [x_g(ix,dx,Ft) for ix=1:ncx];
        yv  = [y_g(iy,dy,Ft) for iy=1:ncy];
        zv  = [z_g(iz,dz,Ft) for iz=1:ncz];
        # Question 2 - how to xv
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

    it1=0; time=0
    transient = 1.0

    ## Action
    for it = it1:nt

        if it==0 
            @printf("\n/******************* Initialisation step *******************\n")
            transient = 0.0
        else
            @printf("\n/******************* Time step %05d *******************\n", it)
            transient = 1.0
        end

        @printf(">>>> Thermal solver\n");
        @parallel UpdateThermalConductivity( ktv, Tc_ex, phv, ϕi, sc.kt, sc.T )
        @parallel SmoothConductivityV2C( kc,  ktv )
        @parallel SmoothConductivityC2V( ktv, kc )
        @parallel InitConductivity!(kx, ky, kz, ktv)
        @parallel ResetA!(Ft, Ft0)
       
        @parallel ComputeρCeffective!(ρC_eff, Tc_ex, Pc_ex, phc, ϕi, sc.T, sc.σ, sc.C, sc.ρ)
        for iter = 1:nitmax
            @parallel SetTemperatureBCs!(Tc_ex, phc, qyS, _dy, 1.0/sc.kt, Ttop,  geometry.y_plateau, geometry.a2, geometry.b2, dTdy, dx, dy, sticky_air)
            @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Tc_ex, _dx, _dy, _dz)
            @parallel ResidualTemperatureLinearised!(Ft, Tc_ex, Tc0, ρC_eff, phc, qx, qy, qz, Qt, transient, _dt, _dx, _dy, _dz)
            # @parallel ResidualTemperatureNonLinear!(Ft, Tc_ex, Tc0, Pc_ex, phc, qx, qy, qz, _dt, _dx, _dy, _dz, ϕi, Qt, transient, sc.T, sc.σ, sc.C, sc.ρ)
            @parallel DampedUpdate!(Ft0, Tc_ex, Ft, dampT, dtauT)
            if (USE_MPI) update_halo!(Tc_ex); end
            if mod(iter,nitout) == 0 || iter==1
                nFt = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz) * (sc.ρ*sc.C*sc.T/sc.t)
                if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
                if (me==0) if isnan(nFt) error("Nan T") end end
                if nFt<tolT break end
            end
        end

        @printf(">>>> Darcy solver\n");
        @parallel ResetA!(Ft, Ft0)
        @parallel UpdateHydroConductivity(kv, kfv, ρfv, Tc_ex, Pc_ex, yv2, phv, sc.σ, sc.T, sc.L, sc.t, sc.ρ, sc.η)
        @parallel SmoothConductivityV2C( kc, kfv )
        @parallel SmoothConductivityC2V( kfv, kc )
        @parallel InitConductivity!(kx, ky, kz, kfv)
        @printf("min(ρfv)   = %11.4e - max(ρfv)   = %11.4e\n", minimum_g(ρfv)*sc.ρ,   maximum_g(ρfv)*sc.ρ )

        # @parallel ϕρfV2C(ϕρfc, ρfv, phv, ϕi)
        # @parallel InitDarcy!(ϕρf0c, ϕρfc)
        @parallel ϕρf(ϕρf0c, Tc_ex, Pc_ex, phc, ϕi, sc.ρ, sc.T, sc.σ)
        @printf("min(ρfc0)   = %11.4e - max(ρfc0)   = %11.4e\n", minimum_g(ϕρf0c)/ϕi,   maximum_g(ϕρf0c)*sc.ρ/ϕi )

        @time for iter = 1:nitmax
            # @parallel UpdateHydroConductivity(kv, kfv, ρfv, Tc_ex, Pc_ex, yv2, phv, sc.σ, sc.T, sc.L, sc.t, sc.ρ, sc.η)
            # @parallel ϕρfV2C(ϕρfc, ρfv, phv, ϕi)
            # @parallel InitConductivity!(kx, ky, kz, kfv)
            # @parallel SmoothConductivityV2C( kc, kfv )
            # @parallel SmoothConductivityC2V( kfv, kc )
            @parallel ϕρf(ϕρfc, Tc_ex, Pc_ex, phc, ϕi, sc.ρ, sc.T, sc.σ)
            @parallel SetPressureBCs!(Pc_ex, phc, Pbot, Ptop, geometry.y_plateau, geometry.a2, geometry.b2, 1000/sc.ρ, g, dx, dy, sticky_air)
            @parallel ComputeDarcyFlux!(qx, qy, qz, ρfv, kx, ky, kz, Pc_ex, g, _dx, _dy, _dz)
            @parallel ResidualFluidPressure!(Ft, phc, ϕρfc, ϕρf0c, qx, qy, qz, transient, _dt, _dx, _dy, _dz)
            @parallel DampedUpdate!(Ft0, Pc_ex, Ft, dampP, dtauP)
            if (USE_MPI) update_halo!(Pc_ex); end
            if mod(iter,nitout) == 0 || iter==1
                nFp = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz) * (sc.ρ/sc.t)
                if (me==0) @printf("PT iter. #%05d - || Fp || = %2.2e\n", iter, nFp) end
                if (me==0) if isnan(nFp) error("Nan P") end end
                if nFp<tolP break end
            end
        end

        # Compute velocity
        @parallel UpdateHydroConductivity(kv, kfv, ρfv, Tc_ex, Pc_ex, yv2, phv, sc.σ, sc.T, sc.L, sc.t, sc.ρ, sc.η)
        @parallel SmoothConductivityV2C( kc, kfv )
        @parallel SmoothConductivityC2V( kfv, kc )
        # @parallel Init_vel!(Vx, Vy, Vz, qx, qy, qz, ρfv, phv)
        @parallel InitConductivity!(kx, ky, kz, kv)
        @parallel ComputeDarcyFlux!(Vx, Vy, Vz, ρfv, kx, ky, kz, Pc_ex, g, _dx, _dy, _dz)


        if it>0
            time  = time + dt;
            @printf("\n-> it=%d, time=%.1e, dt=%.1e, \n", it, time, dt);

            #---------------------------------------------------------------------
            if Advection == 1
                AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Tc0, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, dt, _dx, _dy, _dz, Ttop, Tbot )

                # Set dt for next step
                dt  = dt_fact*1.0/6.1*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
                _dt = 1.0/dt
                @printf("Time step = %2.2e s\n", dt*sc.t)
            end
        end

        @printf("min(Tc_ex) = %11.4e - max(Tc_ex) = %11.4e\n", minimum_g(Tc_ex)*sc.T, maximum_g(Tc_ex)*sc.T )
        @printf("min(Pc_ex) = %11.4e - max(Pc_ex) = %11.4e\n", minimum_g(Pc_ex)*sc.σ, maximum_g(Pc_ex)*sc.σ )
        @printf("min(ρfv)   = %11.4e - max(ρfv)   = %11.4e\n", minimum_g(ρfv)*sc.ρ,   maximum_g(ρfv)*sc.ρ )
        @printf("min(kfv)   = %11.4e - max(kfv)   = %11.4e\n", minimum_g(kfv)*sc.t,   maximum_g(kfv)*sc.t )
        @printf("min(kv)    = %11.4e - max(kv)    = %11.4e\n", minimum_g(kv)*sc.L^2,  maximum_g(kv)*sc.L^2 )
        @printf("min(Vy)    = %11.4e - max(Vy)    = %11.4e\n", minimum_g(Vy)*sc.V,    maximum_g(Vy)*sc.V )

        #---------------------------------------------------------------------
        if (Vizu == 1)
            tMa = @sprintf("%03f", time*sc.t/1e6/year)
            y_topo = Topography.( xce, geometry.y_plateau, geometry.a2, geometry.b2 )
            p1 = heatmap(xce*sc.L/1e3, yce*sc.L/1e3, (Tc_ex[:,:,2]'.*sc.T.-273.15), c=cgrad(:hot, rev=true), aspect_ratio=1, clims=(0, 700), xlim=(0,120), ylim=(-30,5)) 
            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, phv[:,:,2]')
            p2 = heatmap(xce*sc.L/1e3, yce*sc.L/1e3, (Pc_ex[:,:,2]'.*sc.σ./1e6), c=:jet1, aspect_ratio=1, xlim=(0,120), ylim=(-30,5)) 
            p3 = heatmap(xc *sc.L/1e3, yv *sc.L/1e3, (Vy[:,:,2]'.*sc.V*100*year), c=:jet1, aspect_ratio=1, clims=(-27, 23), xlim=(0,120), ylim=(-30,5)) #title="Vy [cm/y]"*string(" @ t = ", tMa, " My" ) 
            p4 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (ρfv[:,:,2]'.*sc.ρ), c=:jet1, aspect_ratio=1, xlim=(0,120), ylim=(-30,5), clims=(550, 1000)) 

            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, phv[:,:,2]', c=:jet1, aspect_ratio=1) 

            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (ktv[:,:,2]'.*sc.kt), c=:jet1, aspect_ratio=1) 
            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (kfv[:,:,2]'.*sc.t), c=:jet1, aspect_ratio=1) 
            p5 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, (kv[:,:,2]'.*sc.L^2)/1e-16, c=:jet1, aspect_ratio=1, xlim=(0,120), ylim=(-30,5)) 


            # p1 = heatmap(xv*sc.L/1e3, yv*sc.L/1e3, log10.(kfv[:,:,2]'.*sc.t), c=:jet1, aspect_ratio=1) 
            # X = Tc_ex[2:end-1,2:end-1,2:end-1]
            #  heatmap(xc, yc, transpose(X[:,:,Int(ceil(ncz/2))]),c=:viridis,aspect_ratio=1) 
            #heatmap(xc*sc.L/1e3, yc*sc.L/1e3, (Ft[:,:,1]'.*(sc.ρ/sc.t)), c=:jet1, aspect_ratio=1) 
            #  heatmap(xv*sc.L/1e3, yv*sc.L/1e3, log10.(kf2[:,:,2]'.*sc.kf), c=:jet1, aspect_ratio=1) 
            #   contourf(xc,yc,transpose(Ty[:,:,Int(ceil(ncz/2))])) ) # accede au sublot 111
            #quiver(x,y,(f,f))
            # p1 = plot!(xce*sc.L/1e3, y_topo*sc.L/1e3, c=:white)
            _, izero =  findmin(abs.(yv))
            # p1 = plot!()
            # p2 = plot!(xlim=(0,120), ylim=(-30,5))
            # p3 = plot!(xlim=(0,120), ylim=(-30,5))
            p6 = plot(xc*sc.L/1e3, Vy[:,izero,2].*sc.V*100*year, label=:none)
            display(plot(p1, p2, p3, p4, layout=(4,1)))
            @printf("Imaged sliced at z index %d over ncx = %d, ncy = %d, ncz = %d --- time is %02f Ma\n", Int(ceil(ncz/2)), ncx, ncy, ncz, time*sc.t/1e6/year)
            #  heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1) 
        end
        #---------------------------------------------------------------------

        if ( Save==1 && mod(it,nout)==0 )
            filename = @sprintf("./HT3DOutput%05d", it)
            vtkfile  = vtk_grid(filename, Array(xc), Array(yc), Array(zc))
            vtkfile["Pressure"]    = Array(Pc_ex[2:end-1,2:end-1,2:end-1])
            vtkfile["Temperature"] = Array(Tc_ex[2:end-1,2:end-1,2:end-1])
            VxC = 0.5*(Vx[2:end,:,:] + Vx[1:end-1,:,:])
            VyC = 0.5*(Vy[:,2:end,:] + Vy[:,1:end-1,:])
            VzC = 0.5*(Vz[:,:,2:end] + Vz[:,:,1:end-1])
            # ktc = 1.0/8.0*(ktv[1:end-1,1:end-1,1:end-1] + ktv[2:end-0,2:end-0,2:end-0] + ktv[2:end-0,1:end-1,1:end-1] + ktv[1:end-1,2:end-0,1:end-1] + ktv[1:end-1,1:end-1,2:end-0] + ktv[1:end-1,2:end-0,2:end-0] + ktv[2:end-0,1:end-1,2:end-0] + ktv[2:end-0,2:end-0,1:end-1])
            # kfc = 1.0/8.0*(kfv[1:end-1,1:end-1,1:end-1] + kfv[2:end-0,2:end-0,2:end-0] + kfv[2:end-0,1:end-1,1:end-1] + kfv[1:end-1,2:end-0,1:end-1] + kfv[1:end-1,1:end-1,2:end-0] + kfv[1:end-1,2:end-0,2:end-0] + kfv[2:end-0,1:end-1,2:end-0] + kfv[2:end-0,2:end-0,1:end-1])
            Vc  = (Array(VxC),Array(VyC),Array(VzC))
            vtkfile["Velocity"] = Vc
            vtkfile["kThermal"] = Array(ktc)
            vtkfile["kHydro"]   = Array(kfc)
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

@time HydroThermal3D()