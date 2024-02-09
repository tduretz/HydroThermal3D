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

include("../tools/Macros.jl")  # Include macros - Cachemisère
include("../tools/Weno5_Routines.jl")

using Printf, Statistics, LinearAlgebra, Plots
# using HDF5
using WriteVTK

############################################## MAIN CODE ##############################################

@views function HydroThermal3D()

    @printf("Starting HydroThermal3D!\n")

    # Visualise
    Advection = 1
    Vizu      = 1
    Save      = 1
    fact      = 1
    nt        = 0
    nout      = 10
    dt_fact   = 10

# Physics
Ra       = 60.0 # important to put the digit there - if not, it is defined as an Int64
dT       = 1.0
Ttop     = 0.0
Tbot     = Ttop + dT
dP       = 0.25
Ptop     = 0.0
Pbot     = Ptop + dP
T_i      = 10.0
rho_cp   = 1.0
lam0     = 1.0
rad      = 1.0
xmin     = -0.0;  xmax =      1.86038; Lx = xmax - xmin
ymin     = -0.0;  ymax = 1.0/4.0*xmax; Ly = ymax - ymin
zmin     = -0.05; zmax =      1.86038; Lz = zmax - zmin
# Numerics
fact     = 4 
ncx      = fact*32-6
ncy      = fact*8 -6
ncz      = 3#fact*32-6
# Preprocessing
if (USE_MPI) me, dims, nprocs, coords, comm = init_global_grid(ncx, ncy, ncz; dimx=2, dimy=2, dimz=2);              #SO: later this can be calles as "me, = ..."
else         me, dims, nprocs, coords       = (0, [1,1,1], 1, [0,0,0]);
end
Nix      = USE_MPI ? nx_g() : ncx                                                #SO: TODO: this is obtained from the global_grid for MPI.
Niy      = USE_MPI ? ny_g() : ncy                                                #SO: this is obtained from the global_grid.
Niz      = USE_MPI ? nz_g() : ncz                                                #SO: this is obtained from the global_grid.
dx, dy, dz = Lx/Nix, Ly/Niy, Lz/Niz                                             #SO: why not dx = Lx/(ncx-1) or dx = Lx/(Nix-1) respectively
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
ktv      =  @ones(ncx+1,ncy+1,ncz+1)
kfv      =  @ones(ncx+1,ncy+1,ncz+1)
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
@time SetInitialConditions(kfv, Tc_ex, xce2, yce2, zce2, xv2, yv2, zv2, yc2, Tbot, Ttop, dT, xmax, ymax, zmax, Ly)
@time Tc_ex = Data.Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE GPU
@time kfv   = Data.Array(kfv)

@printf("Initial conditions were set up!\n")

 #Define kernel launch params (used only if USE_GPU set true).
cuthreads = (32, 8, 1 )
cublocks  = ( 1, 4, 32).*fact

evol=[]; it1=0; time=0             #SO: added warmpup; added one call to tic(); toc(); to get them compiled (to be done differently later).
## Action
for it = it1:nt

    @printf("Thermal solver\n");
	@parallel ResetA!(Ft,Rt)
	@parallel InitThermal!(Rt, kx, ky, kz, Tc_ex, ktv, _dt)

    for iter = 0:nitmax
        @parallel SetPressureBCs!(Tc_ex, Tbot, Ttop)
        @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Tc_ex, _dx, _dy, _dz)
        @parallel UpdateT!(Ft, Tc_ex, Rt, qx, qy, qz, _dt, dtauT, _dx, _dy, _dz)
        if (USE_MPI) update_halo!(Tc_ex); end
        if mod(iter,nitout) == 1
            # nFt = norm(Ft)/sqrt(ncx*ncy*ncz) # Question 3 - how to norm with GPU and MPI
            nFt = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz)
            if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
            if nFt<tolT break end
        end
    end

    @printf("min(Rt)    = %02.4e - max(Rt)    = %02.4e\n", minimum_g(Rt),    maximum_g(Rt)    )
    @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum_g(Tc_ex), maximum_g(Tc_ex) )

    @printf("Darcy solver\n");
	@parallel ResetA!(Ft, Ft0)
	@parallel InitDarcy!(Ty, kx, ky, kz, Tc_ex, kfv, _dt)
    @parallel Compute_Rp!(Rp, ky, Ty, Ra, _dy)

    for iter = 0:nitmax
        @parallel SetPressureBCs!(Pc_ex, Pbot, Ptop)
        @parallel ComputeFlux!(qx, qy, qz, kx, ky, kz, Pc_ex, _dx, _dy, _dz)
        @parallel ResidualDiffusion!(Ft, Rp, qx, qy, qz, _dx, _dy, _dz)
        @parallel UpdateP!(Ft0, Pc_ex, Ft, dampx, dtauP)

        if (USE_MPI) update_halo!(Pc_ex); end
        if mod(iter,nitout) == 1
            nFt = mean_g(abs.(Ft[:]))/sqrt(ncx*ncy*ncz)
            if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
            if nFt<tolP break end
        end
    end

    @printf("min(Rp)    = %02.4e - max(Rp)    = %02.4e\n", minimum_g(Rp),    maximum_g(Rp)    )
    @printf("min(Pc_ex) = %02.4e - max(Pc_ex) = %02.4e\n", minimum_g(Pc_ex), maximum_g(Pc_ex) )

    time  = time + dt;
    @printf("\n-> it=%d, time=%.1e, dt=%.1e, \n", it, time, dt);

    #---------------------------------------------------------------------
    if Advection == 1
        AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Told, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, kx, ky, kz, dt, _dx, _dy, _dz, Ttop, Tbot, Pc_ex, Ty, Ra )
    end

	# Set dt for next step
	dt  = dt_fact*1.0/6.1*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
	_dt = 1/dt
	@printf("Time step = %2.2e s\n", dt)

    #---------------------------------------------------------------------

    if (Vizu == 1)
        X = Tc_ex[2:end-1,2:end-1,2:end-1]
        # display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(ncz/2))]),c=:viridis,aspect_ratio=1) );
        display( heatmap(zc, yc, X[ncx÷2,:,:], c=:jet1, aspect_ratio=1) );

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
if (USE_MPI) finalize_global_grid(); end

end # Diffusion3D()

############################################## Kernels for HT code ##############################################

@parallel function Init_vel!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Pc_ex::Data.Array, Ty::Data.Array, Ra::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(Vx) = -@all(kx) * _dx*@d_xi(Pc_ex)
    @all(Vy) = -@all(ky) *(_dy*@d_yi(Pc_ex) - Ra*@all(Ty))
    @all(Vz) = -@all(kz) * _dz*@d_zi(Pc_ex)

    return
end

@parallel function InitDarcy!(Ty::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Tc_ex::Data.Array, kfv::Data.Array, _dt::Data.Number)

	@all(Ty) = @av_yi(Tc_ex)
	@all(kx) = @av_yza(kfv)
	@all(ky) = @av_xza(kfv)
	@all(kz) = @av_xya(kfv)

	return
end

@parallel function Compute_Rp!(Rp::Data.Array, ky::Data.Array, Ty::Data.Array, Ra::Data.Number, _dy::Data.Number)

    @all(Rp) = Ra*_dy*@dmul_ya(ky, Ty)

    return
end

@parallel function InitThermal!(Rt::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Tc_ex::Data.Array, ktv::Data.Array, _dt::Data.Number)

    @all(Rt) = -_dt * @inn(Tc_ex)
    @all(kx) = @av_yza(ktv)
    @all(ky) = @av_xza(ktv)
    @all(kz) = @av_xya(ktv)

    return
end

@parallel function ComputeFlux!(qx::Data.Array, qy::Data.Array, qz::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, A::Data.Array,
                                _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(qx) = -@all(kx)*(_dx*@d_xi(A))
    @all(qy) = -@all(ky)*(_dy*@d_yi(A))
    @all(qz) = -@all(kz)*(_dz*@d_zi(A))

    return
end

@parallel function UpdateT!(F::Data.Array, T::Data.Array, R::Data.Array, qx::Data.Array, qy::Data.Array, qz::Data.Array, _dt::Data.Number, dtau::Data.Number,
                            _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(F) = @all(R) + _dt*@inn(T) +  _dx*@d_xa(qx) + _dy*@d_ya(qy) + _dz*@d_za(qz)
    @inn(T) = @inn(T) - dtau * @all(F)

    return
end

@parallel function ResidualDiffusion!(F::Data.Array, R::Data.Array, qx::Data.Array, qy::Data.Array, qz::Data.Array,
                                      _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

	@all(F) = @all(R) +  _dx*@d_xa(qx) + _dy*@d_ya(qy) + _dz*@d_za(qz)

    return
end

@parallel function UpdateP!(F0::Data.Array, P::Data.Array, F::Data.Array, dampx::Data.Number, dtau::Data.Number)

    @all(F0) = @all(F) + dampx*@all(F0)
    @inn(P ) = @inn(P) - dtau *@all(F0)

    return
end

@parallel_indices (ix,iy,iz) function SetPressureBCs!(Pc_ex::Data.Array, Pbot::Data.Number, Ptop::Data.Number)

    if (ix==1             && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[1            ,iy,iz] =          Pc_ex[2              ,iy,iz]  end
    if (ix==size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[size(Pc_ex,1),iy,iz] =          Pc_ex[size(Pc_ex,1)-1,iy,iz]  end
    if (ix<=size(Pc_ex,1) && iy==1             && iz<=size(Pc_ex,3)) Pc_ex[ix            ,1,iz] = 2*Pbot - Pc_ex[ix              ,2,iz]  end
    if (ix<=size(Pc_ex,1) && iy==size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[ix,size(Pc_ex,2),iz] = 2*Ptop - Pc_ex[ix,size(Pc_ex,2)-1,iz]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==1            ) Pc_ex[ix,iy,            1] =          Pc_ex[ix,iy,              2]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==size(Pc_ex,3)) Pc_ex[ix,iy,size(Pc_ex,3)] =          Pc_ex[ix,iy,size(Pc_ex,3)-1]  end

    return
end

@views function SetInitialConditions(kfv::Array{Float64,3}, Tc_ex::Array{Float64,3}, xce2::Array{Float64,3}, yce2::Array{Float64,3}, zce2::Array{Float64,3}, xv2::Array{Float64,3}, yv2::Array{Float64,3}, zv2::Array{Float64,3}, yc2::Array{Float64,3}, Tbot::Data.Number, Ttop::Data.Number, dT::Data.Number, xmax::Data.Number, ymax::Data.Number, zmax::Data.Number, Ly::Data.Number)
# @parallel function SetInitialConditions(kfv, Tc_ex, Tbot, Ttop, dT, xce2, yce2, xv2, yv2, Ly, yc2, xmax, ymax)
    # Initial conditions: Draw Fault
    kfv[ yv2.>=2/3*Ly ] .= kfv[ yv2.>=2/3*Ly ] .+ 20
    a = -30*pi/180; b = 0.75
    @. kfv[ (yv2 - xv2*a - b > 0) & (yv2 - xv2*a - (b+0.05) < 0) & (yv2>0.05) & (zv2<1.0)] = 20
    # Thermal field
    @. Tc_ex[2:end-1,2:end-1,2:end-1] = -dT/Ly * yc2 + dT
    # Tc_ex[2:end-1,2:end-1,2:end-1]    = Tc_ex[2:end-1,2:end-1,2:end-1] + rand(ncx,ncy,ncz)/10
    @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:]
    @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:]
    @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1]
    # SET INITIAL THERMAL PERTUBATION
    # @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2 + (zce2-zmax/2)^2) < 0.01 ] += 0.1
end

@views function AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Told, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, kx, ky, kz, dt, _dx, _dy, _dz, Ttop, Tbot, Pc_ex, Ty, Ra )

    @printf("Advecting with Weno5!\n")
    # Advection
    order = 2.0

    @parallel Init_vel!(Vx, Vy, Vz, kx, ky, kz, Pc_ex, Ty, Ra, _dx, _dy, _dz)
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

    # Upwind velocities
    @parallel ResetA!(Vxm, Vxp)
    @parallel VxPlusMinus!(Vxm, Vxp, Vx)

    @parallel ResetA!(Vym, Vyp)
    @parallel VyPlusMinus!(Vym, Vyp, Vy)

    @parallel ResetA!(Vzm, Vzp)
    @parallel VzPlusMinus!(Vzm, Vzp, Vz)

    ########
    @parallel Cpy_inn_to_all!(Tc, Tc_ex)
    ########

    # Advect in x direction
    @parallel ArrayEqualArray!(Told, Tc)
    for io=1:order
        @parallel Boundaries_x_Weno5!(Tc_exxx, Tc, BC_type_W, BC_val_W, BC_type_E, BC_val_E)
        @parallel Gradients_minus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
        @parallel Gradients_plus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
        @parallel Advect!(Tc, Vxp, dTdxm, Vxm, dTdxp, dt)
    end
    @parallel TimeAveraging!(Tc, Told, order)

    # Advect in y direction
    @parallel ArrayEqualArray!(Told, Tc)
    for io=1:order
        @parallel Boundaries_y_Weno5!(Tc_exxx, Tc, BC_type_S, BC_val_S, BC_type_N, BC_val_N)
        @parallel Gradients_minus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
        @parallel Gradients_plus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
        @parallel Advect!(Tc, Vyp, dTdxm, Vym, dTdxp, dt)
    end
    @parallel TimeAveraging!(Tc, Told, order)

    # Advect in z direction
    @parallel ArrayEqualArray!(Told, Tc)
    for io=1:order
        @parallel Boundaries_z_Weno5!(Tc_exxx, Tc, BC_type_B, BC_val_B, BC_type_F, BC_val_F)
        @parallel Gradients_minus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
        @parallel Gradients_plus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
        @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
        @parallel Advect!(Tc, Vzp, dTdxm, Vzm, dTdxp, dt)
    end
    @parallel TimeAveraging!(Tc, Told, order)

    ####
    @parallel Cpy_all_to_inn!(Tc_ex, Tc)
    ###
    @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum(Tc_ex), maximum(Tc_ex) )
end

##################
@time HydroThermal3D()