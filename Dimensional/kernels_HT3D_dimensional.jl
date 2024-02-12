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

@parallel_indices (i,j,k) function ComputeThermalConductivity( ktv, Tce, phv, ϕ0, scale_kt, scale_T )
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

@parallel_indices (i,j,k) function ComputeHydroConductivity(kfv, Tce, Pce, phv, ymin, Δy, k_fact, δ, scale_σ, scale_T, scale_L, scale_t, scale_η, mode)
    if i<=size(kfv, 1) && j<=size(kfv, 2) && k<=size(kfv, 3) 
        # Interpolate from extended centroids to vertices
        Tv  = 1.0/8.0*(Tce[i+1,j+1,k+1] + Tce[i+1,j,k+1] + Tce[i,j+1,k+1] + Tce[i,j,k+1])
        Tv += 1.0/8.0*(Tce[i+1,j+1,k+0] + Tce[i+1,j,k+0] + Tce[i,j+1,k+0] + Tce[i,j,k+0])
        Tv *= scale_T
        Pv  = 1.0/8.0*(Pce[i+1,j+1,k+1] + Pce[i+1,j,k+1] + Pce[i,j+1,k+1] + Pce[i,j,k+1])
        Pv += 1.0/8.0*(Pce[i+1,j+1,k+0] + Pce[i+1,j,k+0] + Pce[i,j+1,k+0] + Pce[i,j,k+0])
        Pv *= scale_σ
        y   = ymin + (j-1)*Δy
        ρk_μ = ρf_C(Tv - 273.15, Pv) * kF(y*scale_L, δ*scale_L) /  μf(Tv)
        if mode == 1
            kfv[i,j,k]  = ρk_μ / scale_t
        elseif mode == 2
            kfv[i,j,k] = (kF(y*scale_L, δ*scale_L) /  μf(Tv)) / (scale_L^2/scale_η)
        elseif mode == 3
            kfv[i,j,k] =  kF(y*scale_L, δ*scale_L) /  scale_L^2
        end
        # Fault
        if phv[i,j,k] == 2.0
            kfv[i,j,k] *= k_fact
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeFluidDensity(ρfv, Tce, Pce, phv, scale_σ, scale_T, scale_ρ)
    if i<=size(ρfv, 1) && j<=size(ρfv, 2) && k<=size(ρfv, 3) 
        # Interpolate from extended centroids to vertices
        Tv  = 1.0/8.0*(Tce[i+1,j+1,k+1] + Tce[i+1,j,k+1] + Tce[i,j+1,k+1] + Tce[i,j,k+1])
        Tv += 1.0/8.0*(Tce[i+1,j+1,k+0] + Tce[i+1,j,k+0] + Tce[i,j+1,k+0] + Tce[i,j,k+0])
        Tv *= scale_T
        Pv  = 1.0/8.0*(Pce[i+1,j+1,k+1] + Pce[i+1,j,k+1] + Pce[i,j+1,k+1] + Pce[i,j,k+1])
        Pv += 1.0/8.0*(Pce[i+1,j+1,k+0] + Pce[i+1,j,k+0] + Pce[i,j+1,k+0] + Pce[i,j,k+0])
        Pv *= scale_σ
        ρfv[i,j,k] = ρf_C(Tv - 273.15, Pv) / scale_ρ
        # Air
        if phv[i,j,k] == 1.0
            ρfv[i,j,k]  = 0.000  / scale_ρ
        end
    end
    return nothing
end

@parallel function Multiply( a, b, c )
    @all(a) = @all(b)*@all(c)
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


@parallel function SwapDYREL!(Xc, Yc, X0c_ex, Y0c )
	@all(Xc) = @inn(X0c_ex)
    @all(Yc) = @all(Y0c)
	return nothing
end

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

@parallel_indices (i,j,k) function ComputeρCeffective!(ρC_eff::Data.Array, Tc_ex::Data.Array, Pc_ex::Data.Array, phc, ϕi, sca_T, sca_σ, sca_C, sca_ρ)
    if i<=size(ρC_eff, 1) && j<=size(ρC_eff, 2) && k<=size(ρC_eff, 3)
        if phc[i,j,k] != 1.0      
            Tsca          = Tc_ex[i+1,j+1,k+1]*sca_T
            Psca          = Pc_ex[i+1,j+1,k+1]*sca_σ
            ρ_eff         = ϕ(phc[i,j,k],ϕi)*ρf_C(Tsca-273.15, Psca) + (1.0-ϕ(phc[i,j,k],ϕi))*ρs(Tsca)
            Ceff          = ϕ(phc[i,j,k],ϕi)*Cf(Tsca)                + (1.0-ϕ(phc[i,j,k],ϕi))*Cs(Tsca)
            ρC_eff[i,j,k] = (ρ_eff*Ceff)/sca_ρ/sca_C
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function ResidualTemperatureLinearised!(F::Data.Array, Tc_ex::Data.Array, Tc_old::Data.Array, ρC_eff, phc, PC, qx, qy, qz, Qt, transient, _dt::Data.Number, _dx, _dy, _dz)
    if i<=size(F, 1) && j<=size(F, 2) && k<=size(F, 3)
        if phc[i,j,k] != 1.0  # if not air
            Q = 0.
            if phc[i,j,k] == 3.0
                Q = Qt
            end         
            F[i,j,k]  = transient*ρC_eff[i,j,k]*_dt*(Tc_ex[i+1,j+1,k+1] - Tc_old[i,j,k]) 
            F[i,j,k] += (qx[i+1,j,k] - qx[i,j,k])*_dx
            F[i,j,k] += (qy[i,j+1,k] - qy[i,j,k])*_dy
            F[i,j,k] += (qz[i,j,k+1] - qz[i,j,k])*_dz
            F[i,j,k] -= Q
            F[i,j,k] /= PC[i,j,k]
        else
            F[i,j,k] = 0.
        end
    end
    return nothing
end

@parallel_indices (ix,iy,iz) function SetTemperatureBCs!(Tc_ex::Data.Array, phc, qyS::Data.Number, _dy, kS, TN::Data.Number, y_plateau, a2, b2, dTdy, Δx, Δy, sticky_air )
    if (ix<size(phc,1) && iy<=size(phc,2) && iz<=size(phc,3))
        if phc[ix,iy,iz] == 1.0
            Tc_ex[ix+1,iy+1,iz+1] = TN
        end
    end
    if (ix==1             && iy<=size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[1            ,iy,iz] =              Tc_ex[2              ,iy,iz]  end
    if (ix==size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[size(Tc_ex,1),iy,iz] =              Tc_ex[size(Tc_ex,1)-1,iy,iz]  end
    if (ix<=size(Tc_ex,1) && iy==1             && iz<=size(Tc_ex,3)) Tc_ex[ix            ,1,iz] = qyS/_dy/kS + Tc_ex[ix              ,2,iz]  end
    if (ix<=size(Tc_ex,1) && iy==size(Tc_ex,2) && iz<=size(Tc_ex,3)) Tc_ex[ix,size(Tc_ex,2),iz] =       2*TN - Tc_ex[ix,size(Tc_ex,2)-1,iz]  end
    if (ix<=size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz==1            ) Tc_ex[ix,iy,            1] =              Tc_ex[ix,iy,              2]  end
    if (ix<=size(Tc_ex,1) && iy<=size(Tc_ex,2) && iz==size(Tc_ex,3)) Tc_ex[ix,iy,size(Tc_ex,3)] =              Tc_ex[ix,iy,size(Tc_ex,3)-1]  end
    return nothing
end

@parallel_indices (i,j,k) function GershgorinPoisson!( G::Data.Array, iPC, ρC_eff::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, transient, dt::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number )
    if i<=size(G, 1) && j<=size(G, 2) && k<=size(G, 3) 
        kW   = kx[i,j,k]
        kE   = kx[i+1,j,k]
        kS   = ky[i,j,k]
        kN   = ky[i,j+1,k]
        kF   = kz[i,j,k+1]
        kB   = kz[i,j,k]
        rhoC = ρC_eff[i,j,k]
        G[i,j,k]    = abs(kE ./ dx .^ 2) + abs(kW ./ dx .^ 2) + abs(kN ./ dy .^ 2) + abs(kS ./ dy .^ 2) + abs(kB ./ dz .^ 2) + abs(kF ./ dz .^ 2) + abs((kB ./ dz + kF ./ dz) ./ dz + (kN ./ dy + kS ./ dy) ./ dy + (kE ./ dx + kW ./ dx) ./ dx + rhoC .* transient ./ dt)
        iPC[i,j,k]  = ((kB ./ dz + kF ./ dz) ./ dz + (kN ./ dy + kS ./ dy) ./ dy + (kE ./ dx + kW ./ dx) ./ dx + rhoC .* transient ./ dt)
        G[i,j,k]   /= ((kB ./ dz + kF ./ dz) ./ dz + (kN ./ dy + kS ./ dy) ./ dy + (kE ./ dx + kW ./ dx) ./ dx + rhoC .* transient ./ dt)
    end
    return nothing
end

######################################### HYDRO #########################################

@parallel_indices (i,j,k) function ϕρf(ϕρfc, Tc_ex, Pc_ex, phc, ϕi, scale_ρ, scale_T, scale_σ)
    if i<=size(ϕρfc, 1) && j<=size(ϕρfc, 2) && k<=size(ϕρfc, 3) 
        if phc[i,j,k] != 1.0 # if not air
            T = Tc_ex[i+1,j+1,k+1]*scale_T - 273.15
            P = Pc_ex[i+1,j+1,k+1]*scale_σ
            ϕρfc[i,j,k] =  ϕ(phc[i,j,k], ϕi) * ρf_C(T, P) / scale_ρ
        end
    end    
    return nothing
end

@parallel_indices (i,j,k) function ϕdρdP!(ϕdρdP, Tc_ex, Pc_ex, phc, ϕi, scale_ρ, scale_T, scale_σ)
    if i<=size(ϕdρdP, 1) && j<=size(ϕdρdP, 2) && k<=size(ϕdρdP, 3) 
        if phc[i,j,k] != 1.0 # if not air
            T = Tc_ex[i+1,j+1,k+1]*scale_T - 273.15
            P = Pc_ex[i+1,j+1,k+1]*scale_σ
            ϕdρdP[i,j,k] =  ϕ(phc[i,j,k], ϕi) * dρdP_C(T, P) / (scale_ρ/scale_σ)
        end
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

@parallel_indices (i,j,k) function ResidualFluidPressure!(F::Data.Array, phc, ϕρfc, ϕρfc0, PC, qx::Data.Array, qy::Data.Array, qz::Data.Array, transient,
    _dt::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)
    if i<=size(F, 1) && j<=size(F, 2) && k<=size(F, 3)
        if phc[i,j,k] != 1.0      
            F[i,j,k]  = transient*_dt*(ϕρfc[i,j,k] - ϕρfc0[i,j,k]) 
            F[i,j,k] += (qx[i+1,j,k] - qx[i,j,k])*_dx
            F[i,j,k] += (qy[i,j+1,k] - qy[i,j,k])*_dy
            F[i,j,k] += (qz[i,j,k+1] - qz[i,j,k])*_dz
            F[i,j,k] /= PC[i,j,k]
        else
            F[i,j,k] = 0.
        end
    end
    return nothing
end

@parallel_indices (ix,iy,iz) function SetPressureBCs!(Pc_ex::Data.Array, phc, PS::Data.Number, PN::Data.Number, y_plateau, a2, b2, ρf, g, Δx, Δy, sticky_air )
    if (ix<size(phc,1) && iy<=size(phc,2) && iz<=size(phc,3))
        if phc[ix,iy,iz] == 1.0
            Pc_ex[ix+1,iy+1,iz+1] = PN
        end
    end
    if (ix==1             && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[1            ,iy,iz] =          Pc_ex[2              ,iy,iz]  end
    if (ix==size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[size(Pc_ex,1),iy,iz] =          Pc_ex[size(Pc_ex,1)-1,iy,iz]  end
    if (ix<=size(Pc_ex,1) && iy==1             && iz<=size(Pc_ex,3)) Pc_ex[ix            ,1,iz] =   2*PS - Pc_ex[ix              ,2,iz]  end
    if (ix<=size(Pc_ex,1) && iy==size(Pc_ex,2) && iz<=size(Pc_ex,3)) Pc_ex[ix,size(Pc_ex,2),iz] =   2*PN - Pc_ex[ix,size(Pc_ex,2)-1,iz]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==1            ) Pc_ex[ix,iy,            1] =          Pc_ex[ix,iy,              2]  end
    if (ix<=size(Pc_ex,1) && iy<=size(Pc_ex,2) && iz==size(Pc_ex,3)) Pc_ex[ix,iy,size(Pc_ex,3)] =          Pc_ex[ix,iy,size(Pc_ex,3)-1]  end
    return nothing
end


@parallel function DampedUpdate!(F0::Data.Array, X::Data.Array, F::Data.Array, dampx::Data.Number, _dτ::Data.Number)
    @all(F0) = @all(F) + dampx*@all(F0)
    @inn(X ) = @inn(X) -   _dτ*@all(F0)
    return nothing
end

@parallel function DYRELUpdate!(F0::Data.Array, X::Data.Array, F::Data.Array, h1::Data.Number, h2::Data.Number, Δτ::Data.Number)
    @all(F0) = h1*@all(F0) + h2*@all(F)
    @inn(X ) = @inn(X)     - Δτ*@all(F0)
    return nothing
end

@parallel function Init_vel!(Vx, Vy, Vz, qx, qy, qz, ρfv)
    @all(Vx) = @all(qx) / @av_yza(ρfv)
    @all(Vy) = @all(qy) / @av_xza(ρfv) 
    @all(Vz) = @all(qz) / @av_xya(ρfv) 
    return nothing
end

@parallel_indices (i,j,k) function GershgorinHydro!( G::Data.Array, ϕdρdP::Data.Array, kx::Data.Array, ky::Data.Array, kz::Data.Array, transient, dt::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number )
    if i<=size(G, 1) && j<=size(G, 2) && k<=size(G, 3) 
        kW   = kx[i,j,k]
        kE   = kx[i+1,j,k]
        kS   = ky[i,j,k]
        kN   = ky[i,j+1,k]
        kF   = kz[i,j,k+1]
        kB   = kz[i,j,k]
        rhoC = ϕdρdP[i,j,k]
        G[i,j,k] = abs(kE ./ dx .^ 2) + abs(kW ./ dx .^ 2) + abs(kN ./ dy .^ 2) + abs(kS ./ dy .^ 2) + abs(kB ./ dz .^ 2) + abs(kF ./ dz .^ 2) + abs((kB ./ dz + kF ./ dz) ./ dz + (kN ./ dy + kS ./ dy) ./ dy + (kE ./ dx + kW ./ dx) ./ dx + + rhoC * transient ./ dt)
    end
    return nothing
end

@parallel_indices (i,j,k) function RayleighQuotientNumerator!( G::Data.Array, Fc0, Fcit, Fc, Δτ )
    if i<=size(G, 1) && j<=size(G, 2) && k<=size(G, 3) 
        δx       = Δτ*Fc0[i,j,k]
        G[i,j,k] = .-δx.*(Fc[i,j,k] .- Fcit[i,j,k])
    end
    return nothing
end

@parallel_indices (i,j,k) function RayleighQuotientDenominator!( G::Data.Array, Fc0, Δτ )
    if i<=size(G, 1) && j<=size(G, 2) && k<=size(G, 3) 
        δx       = Δτ*Fc0[i,j,k]
        G[i,j,k] = δx*δx
    end
    return nothing
end

#################################################
#################################################
#################################################

# @parallel_indices (i,j,k) function ResidualTemperatureNonLinear!(F::Data.Array, Tc_ex::Data.Array, Tc_old::Data.Array, Pc_ex::Data.Array, phc, qx, qy, qz, _dt::Data.Number, _dx, _dy, _dz, ϕi, Qt, transient, sca_T, sca_σ, sca_C, sca_ρ)
#     if i<=size(F, 1) && j<=size(F, 2) && k<=size(F, 3)
#         if phc[i,j,k] != 1.0
#             Q = 0.
#             if phc[i,j,k] == 3.0
#                 Q = Qt
#             end         
#             Tsca       = Tc_ex[i+1,j+1,k+1]*sca_T
#             Psca       = Pc_ex[i+1,j+1,k+1]*sca_σ
#             ρ_eff      = ϕ(phc[i,j,k],ϕi)*ρf(Tsca, Psca) + (1.0-ϕ(phc[i,j,k],ϕi))*ρs(Tsca)
#             Ceff       = ϕ(phc[i,j,k],ϕi)*Cf(Tsca)              + (1.0-ϕ(phc[i,j,k],ϕi))*Cs(Tsca)
#             ρC_eff     = (ρ_eff*Ceff)/sca_ρ/sca_C
#             F[i,j,k]  = transient*ρC_eff*_dt*(Tc_ex[i+1,j+1,k+1] - Tc_old[i,j,k]) 
#             F[i,j,k] += (qx[i+1,j,k] - qx[i,j,k])*_dx
#             F[i,j,k] += (qy[i,j+1,k] - qy[i,j,k])*_dy
#             F[i,j,k] += (qz[i,j,k+1] - qz[i,j,k])*_dz
#             F[i,j,k] -= Q
#         else
#             F[i,j,k] = 0.
#         end
#     end
#     return nothing
# end

# @parallel_indices (i,j,k) function ϕρfV2C(ϕρfc, ρfv, phv, ϕi)
#     if i<=size(ϕρfc, 1) && j<=size(ϕρfc, 2) && k<=size(ϕρfc, 3) 
#         ϕρfc[i,j,k]  = 0.
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+0,k+0], ϕi)*ρfv[i+0,j+0,k+0])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+0,k+0], ϕi)*ρfv[i+1,j+0,k+0])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+1,k+0], ϕi)*ρfv[i+0,j+1,k+0])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+1,k+0], ϕi)*ρfv[i+1,j+1,k+0])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+0,k+1], ϕi)*ρfv[i+0,j+0,k+1])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+0,k+1], ϕi)*ρfv[i+1,j+0,k+1])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+0,j+1,k+1], ϕi)*ρfv[i+0,j+1,k+1])
#         ϕρfc[i,j,k] += 0.125*( ϕ(phv[i+1,j+1,k+1], ϕi)*ρfv[i+1,j+1,k+1])
#     end    
#     return nothing
# end

# @parallel_indices (i,j,k) function Init_vel!(Vx, Vy, Vz, qx, qy, qz, ρfv, phv)
#     if i<=size(Vx, 1) && j<=size(Vx, 2) && k<=size(Vx, 3)
#         scalex = 0.25*(ρfv[i,j,k] + ρfv[i,j+1,k] + ρfv[i,j,k+1] + ρfv[i,j+1,k+1]) 
#         if phv[i,j,k]==1.0 || phv[i,j+1,k]==1.0 || phv[i,j,k+1]==1.0 || phv[i,j+1,k+1]==1.0
#             # scalex = 1e10
#         end
#         Vx[i,j,k] = qx[i,j,k] / scalex
#     end
#     if i<=size(Vy, 1) && j<=size(Vy, 2) && k<=size(Vy, 3)
#         scaley = 0.25*(ρfv[i,j,k] + ρfv[i+1,j,k] + ρfv[i,j,k+1] + ρfv[i+1,j,k+1]) 
#         if phv[i,j,k]==1.0 || phv[i+1,j,k]==1.0 || phv[i,j,k+1]==1.0 || phv[i+1,j,k+1]==1.0
#             # scaley = 1e10
#         end
#         Vy[i,j,k] = qy[i,j,k] / scaley
#     end
#     if i<=size(Vz, 1) && j<=size(Vz, 2) && k<=size(Vz, 3)
#         scalez = 0.25*(ρfv[i,j,k] + ρfv[i+1,j,k] + ρfv[i,j+1,k] + ρfv[i+1,j+1,k]) 
#         if phv[i,j,k]==1.0 || phv[i+1,j,k]==1.0 || phv[i,j+1,k]==1.0 || phv[i+1,j+1,k]==1.0
#             # scalez = 1e10
#         end
#         Vz[i,j,k] = qz[i,j,k] / scalez
#     end
#     return nothing
# end

# @views function AdvectWithWeno5( Tc, Tc_ex, Tc_exxx, Told, dTdxm, dTdxp, Vxm, Vxp, Vym, Vyp, Vzm, Vzp, Vx, Vy, Vz, v1, v2, v3, v4, v5, dt, _dx, _dy, _dz, Ttop, Tbot )

#     @printf("Advecting with Weno5!\n")
#     # Advection
#     order = 2.0

#     # Boundaries
#     BC_type_W = 0
#     BC_val_W  = 0.0
#     BC_type_E = 0
#     BC_val_E  = 0.0

#     BC_type_S = 1
#     BC_val_S  = Tbot
#     BC_type_N = 1
#     BC_val_N  = Ttop

#     BC_type_B = 0
#     BC_val_B  = 0.0
#     BC_type_F = 0
#     BC_val_F  = 0.0

#     # Upwind velocities
#     @parallel ResetA!(Vxm, Vxp)
#     @parallel VxPlusMinus!(Vxm, Vxp, Vx)

#     @parallel ResetA!(Vym, Vyp)
#     @parallel VyPlusMinus!(Vym, Vyp, Vy)

#     @parallel ResetA!(Vzm, Vzp)
#     @parallel VzPlusMinus!(Vzm, Vzp, Vz)

#     ########
#     @parallel Cpy_inn_to_all!(Tc, Tc_ex)
#     ########

#     # Advect in x direction
#     @parallel ArrayEqualArray!(Told, Tc)
#     for io=1:order
#         @parallel Boundaries_x_Weno5!(Tc_exxx, Tc, BC_type_W, BC_val_W, BC_type_E, BC_val_E)
#         @parallel Gradients_minus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
#         @parallel Gradients_plus_x_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
#         @parallel Advect!(Tc, Vxp, dTdxm, Vxm, dTdxp, dt)
#     end
#     @parallel TimeAveraging!(Tc, Told, order)

#     # Advect in y direction
#     @parallel ArrayEqualArray!(Told, Tc)
#     for io=1:order
#         @parallel Boundaries_y_Weno5!(Tc_exxx, Tc, BC_type_S, BC_val_S, BC_type_N, BC_val_N)
#         @parallel Gradients_minus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
#         @parallel Gradients_plus_y_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
#         @parallel Advect!(Tc, Vyp, dTdxm, Vym, dTdxp, dt)
#     end
#     @parallel TimeAveraging!(Tc, Told, order)

#     # Advect in z direction
#     @parallel ArrayEqualArray!(Told, Tc)
#     for io=1:order
#         @parallel Boundaries_z_Weno5!(Tc_exxx, Tc, BC_type_B, BC_val_B, BC_type_F, BC_val_F)
#         @parallel Gradients_minus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxm, v1, v2, v3, v4, v5)
#         @parallel Gradients_plus_z_Weno5!(v1, v2, v3, v4, v5, Tc_exxx, _dx, _dy, _dz)
#         @parallel dFdx_Weno5!(dTdxp, v1, v2, v3, v4, v5)
#         @parallel Advect!(Tc, Vzp, dTdxm, Vzm, dTdxp, dt)
#     end
#     @parallel TimeAveraging!(Tc, Told, order)

#     ####
#     @parallel Cpy_all_to_inn!(Tc_ex, Tc)
#     ###
#     @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum(Tc_ex), maximum(Tc_ex) )
# end