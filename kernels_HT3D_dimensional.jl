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

@parallel function Init_vel!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Pc_ex::Data.Array, Ty::Data.Array, Ra::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(Vx) = -@all(kx) * _dx*@d_xi(Pc_ex)
    @all(Vy) = -@all(ky) *(_dy*@d_yi(Pc_ex) - Ra*@all(Ty))
    @all(Vz) = -@all(kz) * _dz*@d_zi(Pc_ex)

    return nothing
end

# @parallel function InitDarcy!(Ty::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, Tc_ex::Data.Array, kfv::Data.Array, _dt::Data.Number)

# 	@all(Ty) = @av_yi(Tc_ex)
# 	@all(kx) = @av_yza(kfv)
# 	@all(ky) = @av_xza(kfv)
# 	@all(kz) = @av_xya(kfv)

# 	return nothing
# end

@parallel function InitThermal!(Tc0, Tc_ex)
	@all(Tc0) = @inn(Tc_ex)
	return nothing
end

@parallel function InitConductivity!(kx::Data.Array, ky::Data.Array, kz::Data.Array, ktv::Data.Array)
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

@parallel_indices (i,j,k) function ResidualTemperature!(Ft::Data.Array, Tc_ex::Data.Array, Tc_old::Data.Array, Pc_ex::Data.Array, phc, qx, qy, qz, _dt::Data.Number, _dx, _dy, _dz, ϕ0, transient, sca_T, sca_σ, sca_C, sca_ρ)
    if i<=size(Ft, 1) && j<=size(Ft, 2) && k<=size(Ft, 3)
        Tsca       = Tc_ex[i+1,j+1,k+1]*sca_T
        Psca       = Pc_ex[i+1,j+1,k+1]*sca_σ
        ρ_eff      = ϕ(phc[i,j,k],ϕ0)*ρf(Tsca-273.15, Psca) + (1.0-ϕ(phc[i,j,k],ϕ0))*ρs(Tsca)
        Ceff       = ϕ(phc[i,j,k],ϕ0)*Cf(Tsca)              + (1.0-ϕ(phc[i,j,k],ϕ0))*Cs(Tsca)
        ρC_eff     = (ρ_eff*Ceff)/sca_ρ/sca_C
        Ft[i,j,k]  = transient*ρC_eff*_dt*(Tc_ex[i+1,j+1,k+1] - Tc_old[i,j,k]) 
        Ft[i,j,k] += (qx[i+1,j,k] - qx[i,j,k])*_dx
        Ft[i,j,k] += (qy[i,j+1,k] - qy[i,j,k])*_dy
        Ft[i,j,k] += (qz[i,j,k+1] - qz[i,j,k])*_dz
    end
    return nothing
end

@parallel_indices (ix,iy,iz) function SetTemperatureBCs!(Tc_ex::Data.Array, qyS::Data.Number, _dy, kS, TN::Data.Number)

    if (ix==1             && iy<=size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[1            ,iy,iz] =              Tc_ex[2              ,iy,iz]  end
    if (ix==size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[size(Tc_ex,1),iy,iz] =              Tc_ex[size(Tc_ex,1)-1,iy,iz]  end
    if (ix<=size(Tc_ex,1) && iy==1             && iz<=size(Tc_ex,3)) Tc_ex[ix            ,1,iz] = qyS/_dy/kS + Tc_ex[ix              ,2,iz]  end
    # if (ix<=size(Tc_ex,1) && iy==1             && iz<=size(Tc_ex,3)) Tc_ex[ix            ,1,iz] = 2*TN - Tc_ex[ix              ,2,iz]  end
    if (ix<=size(Tc_ex,1) && iy==size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[ix,size(Tc_ex,2),iz] =       2*TN - Tc_ex[ix,size(Tc_ex,2)-1,iz]  end
    if (ix<=size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz==1            ) Tc_ex[ix,iy,            1] =              Tc_ex[ix,iy,              2]  end
    if (ix<=size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz==size(Tc_ex,3)) Tc_ex[ix,iy,size(Tc_ex,3)] =              Tc_ex[ix,iy,size(Tc_ex,3)-1]  end

    return nothing
end

@parallel_indices (i,j,k) function ϕρfV2C(ϕρfc, ρfv, phv, ϕ0)
    if i<=size(ϕρfc, 1) && j<=size(ϕρfc, 2) && k<=size(ϕρfc, 3) 
        ϕρfc[i,j,k]  = 0.
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+0,k+0], ϕ0)*ρfv[i+0,j+0,k+0])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+0,k+0], ϕ0)*ρfv[i+1,j+0,k+0])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+1,k+0], ϕ0)*ρfv[i+0,j+1,k+0])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+1,k+0], ϕ0)*ρfv[i+1,j+1,k+0])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+0,k+1], ϕ0)*ρfv[i+0,j+0,k+1])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+0,k+1], ϕ0)*ρfv[i+1,j+0,k+1])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+1,k+1], ϕ0)*ρfv[i+0,j+1,k+1])
        ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+1,k+1], ϕ0)*ρfv[i+1,j+1,k+1])
    end    
    return nothing
end

@parallel function InitDarcy!(ϕρf0, ϕρf)
	@all(ϕρf0) = @all(ϕρf)
	return nothing
end

@parallel function ComputeDarcyFlux!(qx::Data.Array, qy::Data.Array, qz::Data.Array, ρfv, kx::Data.Array, ky::Data.Array, kz::Data.Array, Pf::Data.Array,
                                g, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)
    @all(qx) = -@all(kx)*(_dx*@d_xi(Pf) )
    @all(qy) = -@all(ky)*(_dy*@d_yi(Pf) - g*@av_xza(ρfv) )
    @all(qz) = -@all(kz)*(_dz*@d_zi(Pf) )
    return nothing
end

@parallel function ResidualFluidPressure!(F::Data.Array, ϕρfc, ϕρfc0, qx::Data.Array, qy::Data.Array, qz::Data.Array,
                                _dt::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)
	@all(F) = _dt*(@all(ϕρfc) -  @all(ϕρfc0)) +  _dx*@d_xa(qx) + _dy*@d_ya(qy) + _dz*@d_za(qz)
    return nothing
end

@parallel function DampedUpdate!(F0::Data.Array, X::Data.Array, F::Data.Array, dampx::Data.Number, _dτ::Data.Number)
    @all(F0) = @all(F) + dampx*@all(F0)
    @inn(X ) = @inn(X) -   _dτ*@all(F0)
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