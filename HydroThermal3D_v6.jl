const USE_GPU  = true
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

include("./tools/Macros.jl")  # Include macros - Cachemisère
include("./tools/Weno5_Routines.jl")

using Printf, Statistics, LinearAlgebra, Plots
# using HDF5
using WriteVTK

# Visualise
Advection = 1
Vizu      = 1
Save      = 1
fact      = 2
nt        = 100
nout      = 10
dt_fact   = 10

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
    # Tc_ex[2:end-1,2:end-1,2:end-1]    = Tc_ex[2:end-1,2:end-1,2:end-1] + rand(nx,ny,nz)/10
    @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:]
    @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:]
    @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1]
    # SET INITIAL THERMAL PERTUBATION
    # @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2 + (zce2-zmax/2)^2) < 0.01 ] += 0.1
end


############################################## MAIN CODE ##############################################

@views function HydroThermal3D()
@printf("Starting HydroThermal3D!\n")
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
fact     = 1 
nx       = fact*32-6
ny       = fact*8 -6
nz       = fact*32-6
# Preprocessing
if (USE_MPI) me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz; dimx=2, dimy=2, dimz=2);              #SO: later this can be calles as "me, = ..."
else         me, dims, nprocs, coords       = (0, [1,1,1], 1, [0,0,0]);
end
Nix      = USE_MPI ? nx_g() : nx                                                #SO: TODO: this is obtained from the global_grid for MPI.
Niy      = USE_MPI ? ny_g() : ny                                                #SO: this is obtained from the global_grid.
Niz      = USE_MPI ? nz_g() : nz                                                #SO: this is obtained from the global_grid.
dx, dy, dz = Lx/Nix, Ly/Niy, Lz/Niz                                             #SO: why not dx = Lx/(nx-1) or dx = Lx/(Nix-1) respectively
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
dampx    = 1*(1-Pdamp/min(nx,ny,nz))
@printf("Go go!!\n")

# Initialisation
Told     = @zeros(nx+0,ny+0,nz+0)
v1       = @zeros(nx+0,ny+0,nz+0)
v2       = @zeros(nx+0,ny+0,nz+0)
v3       = @zeros(nx+0,ny+0,nz+0)
v4       = @zeros(nx+0,ny+0,nz+0)
v5       = @zeros(nx+0,ny+0,nz+0)
dTdxp    = @zeros(nx+0,ny+0,nz+0)
dTdxm    = @zeros(nx+0,ny+0,nz+0)
Tc       = @zeros(nx+0,ny+0,nz+0)
Tc_ex    = @zeros(nx+2,ny+2,nz+2)
Tc_exxx  = @zeros(nx+6,ny+6,nz+6)
Pc_ex    = @zeros(nx+2,ny+2,nz+2)
Ty       =  @ones(nx  ,ny+1,nz  )
ktv      =  @ones(nx+1,ny+1,nz+1)
kfv      =  @ones(nx+1,ny+1,nz+1)
dTdy     = @zeros(nx  ,ny  ,nz  )
Rt       = @zeros(nx  ,ny  ,nz  )
Rp       = @zeros(nx  ,ny  ,nz  )
kx       = @zeros(nx+1,ny  ,nz  )
ky       = @zeros(nx  ,ny+1,nz  )
kz       = @zeros(nx  ,ny  ,nz+1)
qx       = @zeros(nx+1,ny  ,nz  )
qy       = @zeros(nx  ,ny+1,nz  )
qz       = @zeros(nx  ,ny  ,nz+1)
Vx       = @zeros(nx+1,ny  ,nz  )
Vy       = @zeros(nx  ,ny+1,nz  )
Vz       = @zeros(nx  ,ny  ,nz+1)
Ft       = @zeros(nx  ,ny  ,nz  )
Ft0      = @zeros(nx  ,ny  ,nz  )
Vxm      = @zeros(nx+0,ny+0,nz+0)
Vxp      = @zeros(nx+0,ny+0,nz+0)
Vym      = @zeros(nx+0,ny+0,nz+0)
Vyp      = @zeros(nx+0,ny+0,nz+0)
Vzm      = @zeros(nx+0,ny+0,nz+0)
Vzp      = @zeros(nx+0,ny+0,nz+0)

@printf("Memory was allocated!\n")

# Pre-processing
if USE_MPI
    xc  = [x_g(ix,dx,Rt)+dx/2 for ix=1:nx];
    yc  = [y_g(iy,dy,Rt)+dy/2 for iy=1:ny];
    zc  = [z_g(iz,dz,Rt)+dz/2 for iz=1:nz];
    xce = [x_g(ix,dx,Rt)-dx/2 for ix=1:nx+2]; # ACHTUNG
    yce = [y_g(iy,dy,Rt)-dy/2 for iy=1:ny+2]; # ACHTUNG
    zce = [z_g(iz,dz,Rt)-dz/2 for iz=1:nz+2]; # ACHTUNG
    xv  = [x_g(ix,dx,Rt) for ix=1:nx];
    yv  = [y_g(iy,dy,Rt) for iy=1:ny];
    zv  = [z_g(iz,dz,Rt) for iz=1:nz];
    # Question 2 - how to xv
else
    xc  = LinRange(xmin+dx/2, xmax-dx/2, nx)
    yc  = LinRange(ymin+dy/2, ymax-dy/2, ny)
    zc  = LinRange(xmin+dz/2, zmax-dz/2, nz)
    xce = LinRange(xmin-dx/2, xmax+dx/2, nx+2)
    yce = LinRange(ymin-dy/2, ymax+dy/2, ny+2)
    zce = LinRange(xmin-dz/2, zmax+dz/2, nz+2)
    xv  = LinRange(xmin, xmax, nx+1)
    yv  = LinRange(ymin, ymax, ny+1)
    zv  = LinRange(xmin, zmax, nz+1)
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
# cuthreads = (32, 8, 2 )
# cublocks = ceil.(Int, (nx+2, ny+2, nz+2)./cuthreads)
cuthreads = (32, 8, 1 )
cublocks  = ( 1, 4, 32).*fact

evol=[]; it1=0; time=0             #SO: added warmpup; added one call to tic(); toc(); to get them compiled (to be done differently later).
## Action
for it = it1:nt

    @printf("Thermal solver\n");
	@parallel cublocks cuthreads ResetA!(Ft,Rt)
	@parallel cublocks cuthreads InitThermal!(Rt, kx, ky, kz, Tc_ex, ktv, _dt)

    for iter = 0:nitmax
        @parallel cublocks cuthreads SetPressureBCs!(Tc_ex, Tbot, Ttop)
        @parallel cublocks cuthreads ComputeFlux!(qx, qy, qz, kx, ky, kz, Tc_ex, _dx, _dy, _dz)
        @parallel cublocks cuthreads UpdateT!(Ft, Tc_ex, Rt, qx, qy, qz, _dt, dtauT, _dx, _dy, _dz)
        if (USE_MPI) update_halo!(Tc_ex); end
        if mod(iter,nitout) == 1
            # nFt = norm(Ft)/sqrt(nx*ny*nz) # Question 3 - how to norm with GPU and MPI
            nFt = mean_g(abs.(Ft[:]))/sqrt(nx*ny*nz)
            if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
            if nFt<tolT break end
        end
    end

    @printf("min(Rt)    = %02.4e - max(Rt)    = %02.4e\n", minimum_g(Rt),    maximum_g(Rt)    )
    @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum_g(Tc_ex), maximum_g(Tc_ex) )

    @printf("Darcy solver\n");
	@parallel cublocks cuthreads ResetA!(Ft, Ft0)
	@parallel cublocks cuthreads InitDarcy!(Ty, kx, ky, kz, Tc_ex, kfv, _dt)
    @parallel cublocks cuthreads Compute_Rp!(Rp, ky, Ty, Ra, _dy)

    for iter = 0:nitmax
        @parallel cublocks cuthreads SetPressureBCs!(Pc_ex, Pbot, Ptop)
        @parallel cublocks cuthreads ComputeFlux!(qx, qy, qz, kx, ky, kz, Pc_ex, _dx, _dy, _dz)
        @parallel cublocks cuthreads ResidualDiffusion!(Ft, Rp, qx, qy, qz, _dx, _dy, _dz)
        @parallel cublocks cuthreads UpdateP!(Ft0, Pc_ex, Ft, dampx, dtauP)

        if (USE_MPI) update_halo!(Pc_ex); end
        if mod(iter,nitout) == 1
            nFt = mean_g(abs.(Ft[:]))/sqrt(nx*ny*nz)
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
        @printf("Advecting with Weno5!\n")
        # Advection
        order = 2.0

        @parallel cublocks cuthreads Init_vel!(Vx, Vy, Vz, kx, ky, kz, Pc_ex, Ty, Ra, _dx, _dy, _dz)
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
        @parallel cublocks cuthreads ResetA!(Vxm, Vxp)
        @parallel cublocks cuthreads VxPlusMinus!(Vxm, Vxp, Vx)

        @parallel cublocks cuthreads ResetA!(Vym, Vyp)
        @parallel cublocks cuthreads VyPlusMinus!(Vym, Vyp, Vy)

        @parallel cublocks cuthreads ResetA!(Vzm, Vzp)
        @parallel cublocks cuthreads VzPlusMinus!(Vzm, Vzp, Vz)

        ########
        @parallel cublocks cuthreads Cpy_inn_to_all!(Tc, Tc_ex)
        ########

        # Advect in x direction
        @parallel cublocks cuthreads ArrayEqualArray!(Told, Tc)
        for io=1:order
            @parallel cublocks cuthreads Boundaries_x_Weno5!(Tc_exxx, Tc, BC_type_W, BC_val_W, BC_type_E, BC_val_E)
            @parallel cublocks cuthreads Gradients_minus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
            @parallel cublocks cuthreads dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
            @parallel cublocks cuthreads Gradients_plus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
            @parallel cublocks cuthreads dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
            @parallel cublocks cuthreads Advect!(Tc, Vxp, dTdxm, Vxm, dTdxp, dt)
        end
        @parallel cublocks cuthreads TimeAveraging!(Tc, Told, order)

       # Advect in y direction
       @parallel cublocks cuthreads ArrayEqualArray!(Told, Tc)
       for io=1:order
           @parallel cublocks cuthreads Boundaries_y_Weno5!(Tc_exxx, Tc, BC_type_S, BC_val_S, BC_type_N, BC_val_N)
           @parallel cublocks cuthreads Gradients_minus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
           @parallel cublocks cuthreads dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
           @parallel cublocks cuthreads Gradients_plus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
           @parallel cublocks cuthreads dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
           @parallel cublocks cuthreads Advect!(Tc, Vyp, dTdxm, Vym, dTdxp, dt)
       end
       @parallel cublocks cuthreads TimeAveraging!(Tc, Told, order)

       # Advect in z direction
       @parallel cublocks cuthreads ArrayEqualArray!(Told, Tc)
       for io=1:order
           @parallel cublocks cuthreads Boundaries_z_Weno5!(Tc_exxx, Tc, BC_type_B, BC_val_B, BC_type_F, BC_val_F)
           @parallel cublocks cuthreads Gradients_minus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
           @parallel cublocks cuthreads dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
           @parallel cublocks cuthreads Gradients_plus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
           @parallel cublocks cuthreads dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
           @parallel cublocks cuthreads Advect!(Tc, Vzp, dTdxm, Vzm, dTdxp, dt)
       end
       @parallel cublocks cuthreads TimeAveraging!(Tc, Told, order)

	   ####
	   @parallel cublocks cuthreads Cpy_all_to_inn!(Tc_ex, Tc)
	   ###
       @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum(Tc_ex), maximum(Tc_ex) )

    end

	# Set dt for next step
	dt  = dt_fact*1.0/6.1*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
	_dt = 1/dt
	@printf("Time step = %2.2e s\n", dt)

    #---------------------------------------------------------------------

    if (Vizu == 1)
        X = Tc_ex[2:end-1,2:end-1,2:end-1]
        # display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(nz/2))]),c=:viridis,aspect_ratio=1) );
        display( heatmap(zc, yc, (X[Int(ceil(nx/2)),:,:]),c=:viridis,aspect_ratio=1) );

        # display(  contourf(xc,yc,transpose(Ty[:,:,Int(ceil(nz/2))])) ) # accede au sublot 111
        #quiver(x,y,(f,f))
        @printf("Imaged sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
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
        # filename = @sprintf("./HT3DOutput%05d.h5", it)
        # if isfile(filename)==1
        #     rm(filename)
        # end
        # h5write(filename, "Tc", Array(Tc_ex))
        # h5write(filename, "Pc", Array(Pc_ex))
        # h5write(filename, "xc", Array(xce))
        # h5write(filename, "yc", Array(yce))
        # h5write(filename, "zc", Array(zce))
    end

end#it
#
# if me == 0
#     # println("time_s=$time_s T_eff=$T_eff niter=$niter iterMin=$iterMin iterMax=$iterMax");
#     # println("nprocs $nprocs dims $(dims[1]) $(dims[2]) $(dims[3]) fdims 1 1 1 nxyz $nx $ny $nz nt $(iterMin-warmup) nb_it 1 PRECIS $(sizeof(Data.Number)) time_s $time_s block $(cuthreads[1]) $(cuthreads[2]) $(cuthreads[3]) grid $(cublocks[1]) $(cublocks[2]) $(cublocks[3])\n");
#     gif(anim, "Diffusion_fps15_1.gif", fps=15);
# end
if (USE_MPI) finalize_global_grid(); end

# =#

end # Diffusion3D()

@time HydroThermal3D()
