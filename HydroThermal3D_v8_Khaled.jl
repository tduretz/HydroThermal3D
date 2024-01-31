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

include("./tools/Macros.jl")  # Include macros - Cachemisère
include("./tools/Weno5_Routines.jl")
include("./kernels_HT3D.jl")

############################################## MAIN CODE ##############################################

@views function HydroThermal3D()

    @printf("Starting HydroThermal3D!\n")

    # Visualise
    Advection = 1
    Vizu      = 1
    Save      = 0
    fact      = 1
    nt        = 0
    nout      = 10
    dt_fact   = 10

    # Characteristic dimensions
    sc   = scaling()
    sc.T = 500.0
    sc.V = 1e-9
    sc.L = 100e3
    sc.η = 1e10
    scale_me!( sc )

    # kfc = 5e-15*exp((y - 500)/3000)                 OK
    # kff = Perm*kc                                   OK

    # Physics
    xmin     = -0.0/sc.L;  xmax = 120.0e3/sc.L; Lx = xmax - xmin
    ymin     = -30e3/sc.L; ymax = 0.0/sc.L;     Ly = ymax - ymin
    zmin     = -0.00/sc.L; zmax = 1.0e3/sc.L;   Lz = zmax - zmin
    dT       = (ymax-ymin)*(30e-3/(sc.T/sc.L))
    Ttop     = 293.0/sc.T
    Tbot     = Ttop + dT
    dP       = 0.25/sc.σ
    Ptop     = 0.0/sc.σ
    Pbot     = Ptop + dP
    # T_i      = 10.0
    # rho_cp   = 1.0
    # lam0     = 1.0
    # rad      = 1.0
    Perm     = 50.0
    ϵp       = 0.1
    θs       = 1.0 - ϵp

    @printf("Surface T = %03f, bottom T = %03f\n", Ttop*sc.T, Tbot*sc.T)

    # Numerics
    fact     = 8 
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
    dt       = min(dx,dy,dz)^2/6.1
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    _dt      = 1.0/dt
    # PT iteration parameters
    nitmax  = 1e4
    nitout  = 1000
    # Thermal solver
    tolT    = 1e-8
    tetT    = 0.1
    dtauT   = tetT*min(dx,dy,dz)^2/4.1
    # Darcy solver
    tolP     = 1e-10
    tetP     = 1/4/3
    dtauP    = tetP/6.1*min(dx,dy,dz)^2
    Pdamp    = 1.25
    dampx    = 1*(1-Pdamp/min(ncx,ncy,ncz))
    @printf("Go go!!\n")

    # Initialisation
    Told     = @zeros(ncx+0,ncy+0,ncz+0)
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
    Ty       =  @ones(ncx  ,ncy+1,ncz  )
    kfv      =  @ones(ncx+1,ncy+1,ncz+1)
    kts      =  @ones(ncx+1,ncy+1,ncz+1)
    kff      =  @ones(ncx+1,ncy+1,ncz+1)
    Cps      =  @ones(ncx+0,ncy+0,ncz+0)
    Cpf      =  @ones(ncx+0,ncy+0,ncz+0)
    μf       =  @ones(ncx+1,ncy+1,ncz+1)
    dTdy     = @zeros(ncx  ,ncy  ,ncz  )
    Rt       = @zeros(ncx  ,ncy  ,ncz  )
    Rp       = @zeros(ncx  ,ncy  ,ncz  )
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

    @printf("Memory was allocated!\n")

    # Pre-processing
    if USE_MPI
        xc  = [x_g(ix,dx,Rt)+dx/2 for ix=1:ncx];
        yc  = [y_g(iy,dy,Rt)+dy/2 for iy=1:ncy];
        zc  = [z_g(iz,dz,Rt)+dz/2 for iz=1:ncz];
        xce = [x_g(ix,dx,Rt)-dx/2 for ix=1:ncx+2]; # ACHTUNG
        yce = [y_g(iy,dy,Rt)-dy/2 for iy=1:ncy+2]; # ACHTUNG
        zce = [z_g(iz,dz,Rt)-dz/2 for iz=1:ncz+2]; # ACHTUNG
        xv  = [x_g(ix,dx,Rt) for ix=1:ncx];
        yv  = [y_g(iy,dy,Rt) for iy=1:ncy];
        zv  = [z_g(iz,dz,Rt) for iz=1:ncz];
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
    (xc2,yc2,zc2)    = ([x for x=xc,y=yc,z=zc],    [y for x=xc,y=yc,z=zc],    [z for x=xc,y=yc,z=zc]   ) #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
    (xv2,yv2,zv2)    = ([x for x=xv,y=yv,z=zv],    [y for x=xv,y=yv,z=zv],    [z for x=xv,y=yv,z=zv]   )
    @printf("Grid was set up!\n")

    @time Tc_ex = Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE CPU:
    @time kfv   = Array(kfv)   # Ensure it is temporarily a CPU array
    @time SetInitialConditions(kfv, Tc_ex, xce2, yce2, zce2, xv2, yv2, zv2, yc2, Tbot, Ttop, dT, xmax, ymax, zmax, Ly, Perm, sc)
    @time Tc_ex = Data.Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE GPU
    @time kfv   = Data.Array(kfv)

    @printf("Initial conditions were set up!\n")

    # #Define kernel launch params (used only if USE_GPU set true).
    # cuthreads = (32, 8, 1 )
    # cublocks  = ( 1, 4, 32).*fact

    # evol=[]; it1=0; time=0             #SO: added warmpup; added one call to tic(); toc(); to get them compiled (to be done differently later).
    # ## Action
    # for it = it1:nt

    #     @printf("Thermal solver\n");
    # 	@parallel ResetA!(Ft,Rt)
    # 	@parallel InitThermal!(Rt, kx, ky, kz, Tc_ex, ktv, _dt)

    #     for iter = 0:nitmax
    #         @parallel SetPressureBCs!(Tc_ex, Tbot, Ttop)
    #         @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Tc_ex, _dx, _dy, _dz)
    #         @parallel UpdateT!(Ft, Tc_ex, Rt, qx, qy, qz, _dt, dtauT, _dx, _dy, _dz)
    #         if (USE_MPI) update_halo!(Tc_ex); end
    #         if mod(iter,nitout) == 1
    #             # nFt = norm(Ft)/sqrt(ncx*ncy*ncz) # Question 3 - how to norm with GPU and MPI
    #             nFt = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz)
    #             if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
    #             if nFt<tolT break end
    #         end
    #     end

    #     @printf("min(Rt)    = %02.4e - max(Rt)    = %02.4e\n", minimum_g(Rt),    maximum_g(Rt)    )
    #     @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum_g(Tc_ex), maximum_g(Tc_ex) )

    #     @printf("Darcy solver\n");
    # 	@parallel ResetA!(Ft, Ft0)
    # 	@parallel InitDarcy!(Ty, kx, ky, kz, Tc_ex, kfv, _dt)
    #     @parallel Compute_Rp!(Rp, ky, Ty, Ra, _dy)

    #     for iter = 0:nitmax
    #         @parallel SetPressureBCs!(Pc_ex, Pbot, Ptop)
    #         @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Pc_ex, _dx, _dy, _dz)
    #         @parallel ResidualDiffusion!(Ft, Rp, qx, qy, qz, _dx, _dy, _dz)
    #         @parallel UpdateP!(Ft0, Pc_ex, Ft, dampx, dtauP)

    #         if (USE_MPI) update_halo!(Pc_ex); end
    #         if mod(iter,nitout) == 1
    #             nFt = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz)
    #             if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
    #             if nFt<tolP break end
    #         end
    #     end

    #     @printf("min(Rp)    = %02.4e - max(Rp)    = %02.4e\n", minimum_g(Rp),    maximum_g(Rp)    )
    #     @printf("min(Pc_ex) = %02.4e - max(Pc_ex) = %02.4e\n", minimum_g(Pc_ex), maximum_g(Pc_ex) )

    #     time  = time + dt;
    #     @printf("\n-> it=%d, time=%.1e, dt=%.1e, \n", it, time, dt);

    #     #---------------------------------------------------------------------
    #     if Advection == 1
    #         AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Told, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, kx, ky, kz, dt, _dx, _dy, _dz, Ttop, Tbot, Pc_ex, Ty, Ra )
    #     end

    # 	# Set dt for next step
    # 	dt  = dt_fact*1.0/6.1*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
    # 	_dt = 1/dt
    # 	@printf("Time step = %2.2e s\n", dt)

    #---------------------------------------------------------------------

    if (Vizu == 1)
        X = Tc_ex[2:end-1,2:end-1,2:end-1]
        # display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(ncz/2))]),c=:viridis,aspect_ratio=1) );
        display( heatmap(xce*sc.L/1e3, yce*sc.L/1e3, Tc_ex[:,:,2]'.*sc.T.-273.15, c=:jet1, aspect_ratio=1) );
        # display( heatmap(xv*sc.L/1e3, yv*sc.L/1e3, log10.(kfv[:,:,1]'.*sc.kf), c=:jet1, aspect_ratio=1) );
        # display(  contourf(xc,yc,transpose(Ty[:,:,Int(ceil(ncz/2))])) ) # accede au sublot 111
        #quiver(x,y,(f,f))
        @printf("Imaged sliced at z index %d over ncx = %d, ncy = %d, ncz = %d\n", Int(ceil(ncz/2)), ncx, ncy, ncz)
        # display( heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1) );
    end

    if ( Save==1 && mod(it,nout)==0 )
        filename = @sprintf("./HT3DOutput%05d", it)
        vtkfile  = vtk_grid(filename, Array(xc), Array(yc), Array(zc))
        vtkfile["Pressure"]    = Array(Pc_ex[2:end-1,2:end-1,2:end-1])
        vtkfile["Temperature"] = Array(Tc_ex[2:end-1,2:end-1,2:end-1])
        VxC = 0.5*(Vx[2:end,:,:]+Vx[1:end-1,:,:])
        VyC = 0.5*(Vy[:,2:end,:]+Vy[:,1:end-1,:])
        VzC = 0.5*(Vz[:,:,2:end]+Vz[:,:,1:end-1])
        ktc = 1.0/8.0*(ktv[1:end-1,1:end-1,1:end-1] + ktv[2:end-0,2:end-0,2:end-0] + ktv[2:end-0,1:end-1,1:end-1] + ktv[1:end-1,2:end-0,1:end-1] + ktv[1:end-1,1:end-1,2:end-0] + ktv[1:end-1,2:end-0,2:end-0] + ktv[2:end-0,1:end-1,2:end-0] + ktv[2:end-0,2:end-0,1:end-1])
        kfc = 1.0/8.0*(kfv[1:end-1,1:end-1,1:end-1] + kfv[2:end-0,2:end-0,2:end-0] + kfv[2:end-0,1:end-1,1:end-1] + kfv[1:end-1,2:end-0,1:end-1] + kfv[1:end-1,1:end-1,2:end-0] + kfv[1:end-1,2:end-0,2:end-0] + kfv[2:end-0,1:end-1,2:end-0] + kfv[2:end-0,2:end-0,1:end-1])
        Vc  = (Array(VxC),Array(VyC),Array(VzC))
        vtkfile["Velocity"] = Vc
        vtkfile["kThermal"] = Array(ktc)
        vtkfile["kHydro"]   = Array(kfc)
        outfiles = vtk_save(vtkfile)
    end

end#it
#
# if me == 0
#     # println("time_s=$time_s T_eff=$T_eff niter=$niter iterMin=$iterMin iterMax=$iterMax");
#     # println("nprocs $nprocs dims $(dims[1]) $(dims[2]) $(dims[3]) fdims 1 1 1 nxyz $ncx $ncy $ncz nt $(iterMin-warmup) nb_it 1 PRECIS $(sizeof(Data.Number)) time_s $time_s block $(cuthreads[1]) $(cuthreads[2]) $(cuthreads[3]) grid $(cublocks[1]) $(cublocks[2]) $(cublocks[3])\n");
#     gif(anim, "Diffusion_fps15_1.gif", fps=15);
# end
# if (USE_MPI) finalize_global_grid(); end

# end 

@time HydroThermal3D()