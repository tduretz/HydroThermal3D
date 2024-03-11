@parallel_indices (i,j,k) function Phase!( X, phc_ex )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            X[i,j,k] = phc_ex[i+1,j+1,k+1]
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Pressure!( X, Pc_ex, phc_ex, scale_σ )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            X[i,j,k] = Pc_ex[i+1,j+1,k+1] * scale_σ
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Temperature!( X, Tc_ex, phc_ex, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            X[i,j,k] = Tc_ex[i+1,j+1,k+1] * scale_T
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Velocity!( X, Y, Z, Vx, Vy, Vz, phc_ex, scale_V )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            X[i,j,k] = 0.5*(Vx[i,j,k] + Vx[i+1,j,k]) * scale_V
            Y[i,j,k] = 0.5*(Vy[i,j,k] + Vy[i,j+1,k]) * scale_V
            Z[i,j,k] = 0.5*(Vz[i,j,k] + Vz[i,j,k+1]) * scale_V
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function EffectiveThermalConductivity!( X, Tc_ex, ϕ0, phc_ex, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            T        = Tc_ex[i+1,j+1,k+1] * scale_T
            X[i,j,k] = ((1.0 - ϕ(phc_ex[i+1,j+1,k+1],ϕ0))*kTs(T) + ϕ(phc_ex[i+1,j+1,k+1],ϕ0)*kTf(T))
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end


@parallel_indices (i,j,k) function EffectiveHeatCapacity!( Ceff::Data.Array, Tc_ex::Data.Array, Pc_ex::Data.Array, phc_ex, ϕi, sca_T, sca_σ, sca_C, sca_ρ)
    if i<=size(Ceff, 1) && j<=size(Ceff, 2) && k<=size(Ceff, 3)
        if phc_ex[i+1,j+1,k+1] != 1.0      
            Tsca          = Tc_ex[i+1,j+1,k+1]*sca_T
            # Ceff[i,j,k]   = ϕ(phc_ex[i+1,j+1,k+1],ϕi)*Cf(Tsca)                + (1.0-ϕ(phc_ex[i+1,j+1,k+1],ϕi))*Cs(Tsca)
            Ceff[i,j,k]   = Cs(Tsca)

        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Permeability!( X, k_fact, ymin, Δy, phc_ex, δ, scale_L )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1.0
            y        = ymin + (j-1)*Δy + Δy/2
            X[i,j,k] = kF(y*scale_L, δ*scale_L)
            if phc_ex[i+1,j+1,k+1] == 2.0
                X[i,j,k] *= k_fact
            end
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function FluidDensity!( X, Tc_ex, Pc_ex, phc_ex, scale_T, scale_σ )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            T        = Tc_ex[i+1,j+1,k+1]*scale_T - 273.15
            P        = Pc_ex[i+1,j+1,k+1]*scale_σ
            X[i,j,k] = ρf_C(T, P)
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Viscosity!( X, Tc_ex, phc_ex, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            T        = Tc_ex[i+1,j+1,k+1]*scale_T
            X[i,j,k] = μf(T)
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Peclet!( X, Tc_ex, Pc_ex, Vx, Vy, Vz, phc_ex, ϕ0, scale_T, scale_σ, scale_V, scale_L )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc_ex[i+1,j+1,k+1] != 1
            x        = 0.5*(Vx[i,j,k] + Vx[i+1,j,k]) * scale_V
            y        = 0.5*(Vy[i,j,k] + Vy[i,j+1,k]) * scale_V
            z        = 0.5*(Vz[i,j,k] + Vz[i,j,k+1]) * scale_V
            V        = sqrt(x^2 + y^2 + z^2)
            T        = Tc_ex[i+1,j+1,k+1]*scale_T
            P        = Pc_ex[i+1,j+1,k+1]*scale_σ
            ρ        = ρf_C(T - 273.15, P)
            C        = Cf(T)
            keff     = ((1-ϕ(phc_ex[i+1,j+1,k+1],ϕ0))*kTs(T) + ϕ(phc_ex[i+1,j+1,k+1],ϕ0)*kTf(T))
            X[i,j,k] = scale_L*ρ*C*V / keff
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end