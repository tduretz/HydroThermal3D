@parallel_indices (i,j,k) function Phase!( X, phc )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            X[i,j,k] = phc[i,j,k]
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Pressure!( X, Pc_ex, phc, scale_σ )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            X[i,j,k] = Pc_ex[i+1,j+1,k+1] * scale_σ
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Temperature!( X, Tc_ex, phc, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            X[i,j,k] = Tc_ex[i+1,j+1,k+1] * scale_T
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Velocity!( X, Y, Z, Vx, Vy, Vz, phc, scale_V )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            X[i,j,k] = 0.5*(Vx[i,j,k] + Vx[i+1,j,k]) * scale_V
            Y[i,j,k] = 0.5*(Vy[i,j,k] + Vy[i,j+1,k]) * scale_V
            Z[i,j,k] = 0.5*(Vz[i,j,k] + Vz[i,j,k+1]) * scale_V
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function EffectiveThermalConductivity!( X, Tc_ex, ϕ0, phc, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            T        = Tc_ex[i+1,j+1,k+1] * scale_T
            X[i,j,k] = ((1.0 - ϕ(phc[i,j,k],ϕ0))*kTs(T) + ϕ(phc[i,j,k],ϕ0)*kTf(T))
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Permeability!( X, ymin, Δy, phc, δ, scale_L )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            y        = ymin + (j-1)*Δy + Δy/2
            X[i,j,k] = kF(y*scale_L, δ*scale_L)
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function FluidDensity!( X, Tc_ex, Pc_ex, phc, scale_T, scale_σ )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            T        = Tc_ex[i+1,j+1,k+1]*scale_T - 273.15
            P        = Pc_ex[i+1,j+1,k+1]*scale_σ
            X[i,j,k] = ρf_C(T, P)
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Viscosity!( X, Tc_ex, phc, scale_T )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            T        = Tc_ex[i+1,j+1,k+1]*scale_T
            X[i,j,k] = μf(T)
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end

@parallel_indices (i,j,k) function Peclet!( X, Tc_ex, Pc_ex, Vx, Vy, Vz, phc, ϕ0, scale_T, scale_σ, scale_V, scale_L )
    if i<=size(X, 1) && j<=size(X, 2) && k<=size(X, 3) 
        if phc[i,j,k] != 1
            x        = 0.5*(Vx[i,j,k] + Vx[i+1,j,k]) * scale_V
            y        = 0.5*(Vy[i,j,k] + Vy[i,j+1,k]) * scale_V
            z        = 0.5*(Vz[i,j,k] + Vz[i,j,k+1]) * scale_V
            V        = sqrt(x^2 + y^2 + z^2)
            T        = Tc_ex[i+1,j+1,k+1]*scale_T
            P        = Pc_ex[i+1,j+1,k+1]*scale_σ
            ρ        = ρf_C(T - 273.15, P)
            C        = Cf(T)
            keff     = ((1-ϕ(phc[i,j,k],ϕ0))*kTs(T) + ϕ(phc[i,j,k],ϕ0)*kTf(T))
            X[i,j,k] = scale_L*ρ*C*V / keff
        else
            X[i,j,k] = NaN
        end
    end
    return nothing
end