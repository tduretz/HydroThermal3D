############################################## Scaling ##############################################

Base.@kwdef mutable struct scaling
    T ::Union{Data.Number, Missing} = missing
    V ::Union{Data.Number, Missing} = missing
    L ::Union{Data.Number, Missing} = missing
    η ::Union{Data.Number, Missing} = missing
    t ::Union{Data.Number, Missing} = missing
    a ::Union{Data.Number, Missing} = missing
    ε ::Union{Data.Number, Missing} = missing
    σ ::Union{Data.Number, Missing} = missing
    m ::Union{Data.Number, Missing} = missing
    ρ ::Union{Data.Number, Missing} = missing
    F ::Union{Data.Number, Missing} = missing
    J ::Union{Data.Number, Missing} = missing
    W ::Union{Data.Number, Missing} = missing
    C ::Union{Data.Number, Missing} = missing
    kt::Union{Data.Number, Missing} = missing
    kf::Union{Data.Number, Missing} = missing
end

function scale_me!( scale )
    scale.t    = scale.L / scale.V;
    scale.a    = scale.V / scale.t;
    scale.ε    = 1.0 / scale.t;
    scale.σ    = scale.η / scale.t
    scale.m    = scale.σ * scale.L * scale.t^2
    scale.ρ    = scale.m / scale.L^3
    scale.F    = scale.m * scale.L / scale.t^2
    scale.J    = scale.m * scale.L^2 / scale.t^2
    scale.W    = scale.J / scale.t
    scale.C    = scale.J / scale.m / scale.T
    scale.kt   = scale.W / scale.L / scale.T
    scale.kf   = scale.L^2
    return nothing
end

############################################## Kernels for HT code ##############################################

@views function SetInitialConditions(kfv::Array{Float64,3}, Tc_ex::Array{Float64,3}, xce2::Array{Float64,3}, yce2::Array{Float64,3}, zce2::Array{Float64,3}, xv2::Array{Float64,3}, yv2::Array{Float64,3}, zv2::Array{Float64,3}, yc2::Array{Float64,3}, Tbot::Data.Number, Ttop::Data.Number, dT::Data.Number, xmax::Data.Number, ymax::Data.Number, zmax::Data.Number, Ly::Data.Number, Perm::Data.Number, sc::scaling)
        # Initial conditions: Draw Fault
        in_fault = zero(kfv)
        # top
        x1 = 68466.18089117216/sc.L; x2 = 31498.437753425103/sc.L
        y1 = 0.0/sc.L;               y2 = -16897.187956165293/sc.L
        a1 = ( y1-y2 ) / ( x1-x2 )
        b1 = y1 - x1*a1
        # bottom
        x1 = 32000.0/sc.L; x2 = 69397.54576150524/sc.L
        y1 =-17800.0/sc.L; y2 = 0.0/sc.L
        a2 = ( y1-y2 ) / ( x1-x2 )
        b2 = y1 - x1*a2
        # bottom
        x1 = 32000.0/sc.L; x2 = 31498.437753425103/sc.L
        y1 =-17800.0/sc.L; y2 = -16897.187956165293/sc.L
        a3 = ( y1-y2 ) / ( x1-x2 )
        b3 = y1 - x1*a3
        kf0 = 5e-15/sc.kf
        y1  = 500.0/sc.L; y2 = 3000.0/sc.L
        @. kfv = kf0*exp((yv2 - y1)/y2)
        @. in_fault = yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)
        @. kfv[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ] = Perm*kfv[ yv2 < (xv2*a1 + b1) && yv2 > (xv2*a2 + b2) && yv2 > (xv2*a3 + b3)  ]
        # Thermal field
        @. Tc_ex = Ttop - dT/Ly * yce2
        @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:]
        @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:]
        @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1]
        # SET INITIAL THERMAL PERTUBATION
        # @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2 + (zce2-zmax/2)^2) < 0.01 ] += 0.1
        return nothing
end

@parallel function Init_vel!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Pc_ex::Data.Array, Ty::Data.Array, Ra::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(Vx) = -@all(kx) * _dx*@d_xi(Pc_ex)
    @all(Vy) = -@all(ky) *(_dy*@d_yi(Pc_ex) - Ra*@all(Ty))
    @all(Vz) = -@all(kz) * _dz*@d_zi(Pc_ex)

    return nothing
end

@parallel function InitDarcy!(Ty::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Tc_ex::Data.Array, kfv::Data.Array, _dt::Data.Number)

	@all(Ty) = @av_yi(Tc_ex)
	@all(kx) = @av_yza(kfv)
	@all(ky) = @av_xza(kfv)
	@all(kz) = @av_xya(kfv)

	return nothing
end

@parallel function Compute_Rp!(Rp::Data.Array, ky::Data.Array, Ty::Data.Array, Ra::Data.Number, _dy::Data.Number)

    @all(Rp) = Ra*_dy*@dmul_ya(ky, Ty)

    return nothing
end

@parallel function InitThermal!(Rt::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Tc_ex::Data.Array, ktv::Data.Array, _dt::Data.Number)

    @all(Rt) = -_dt * @inn(Tc_ex)
    @all(kx) = @av_yza(ktv)
    @all(ky) = @av_xza(ktv)
    @all(kz) = @av_xya(ktv)

    return nothing
end

@parallel function ComputeFlux!(qx::Data.Array, qy::Data.Array, qz::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, A::Data.Array,
                                _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(qx) = -@all(kx)*(_dx*@d_xi(A))
    @all(qy) = -@all(ky)*(_dy*@d_yi(A))
    @all(qz) = -@all(kz)*(_dz*@d_zi(A))

    return nothing
end

@parallel function UpdateT!(F::Data.Array, T::Data.Array, R::Data.Array, qx::Data.Array, qy::Data.Array, qz::Data.Array, _dt::Data.Number, dtau::Data.Number,
                            _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(F) = @all(R) + _dt*@inn(T) +  _dx*@d_xa(qx) + _dy*@d_ya(qy) + _dz*@d_za(qz)
    @inn(T) = @inn(T) - dtau * @all(F)

    return nothing
end

@parallel function ResidualDiffusion!(F::Data.Array, R::Data.Array, qx::Data.Array, qy::Data.Array, qz::Data.Array,
                                      _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

	@all(F) = @all(R) +  _dx*@d_xa(qx) + _dy*@d_ya(qy) + _dz*@d_za(qz)

    return nothing
end

@parallel function UpdateP!(F0::Data.Array, P::Data.Array, F::Data.Array, dampx::Data.Number, dtau::Data.Number)

    @all(F0) = @all(F) + dampx*@all(F0)
    @inn(P ) = @inn(P) - dtau *@all(F0)

    return nothing
end

@parallel_indices (ix,iy,iz) function SetPressureBCs!(Pc_ex::Data.Array, Pbot::Data.Number, Ptop::Data.Number)

    if (ix==1             && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[1            ,iy,iz] =          Pc_ex[2              ,iy,iz]  end
    if (ix==size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[size(Pc_ex,1),iy,iz] =          Pc_ex[size(Pc_ex,1)-1,iy,iz]  end
    if (ix<=size(Pc_ex,1) && iy==1             && iz<=size(Pc_ex,3)) Pc_ex[ix            ,1,iz] = 2*Pbot - Pc_ex[ix              ,2,iz]  end
    if (ix<=size(Pc_ex,1) && iy==size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[ix,size(Pc_ex,2),iz] = 2*Ptop - Pc_ex[ix,size(Pc_ex,2)-1,iz]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==1            ) Pc_ex[ix,iy,            1] =          Pc_ex[ix,iy,              2]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==size(Pc_ex,3)) Pc_ex[ix,iy,size(Pc_ex,3)] =          Pc_ex[ix,iy,size(Pc_ex,3)-1]  end

    return nothing
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