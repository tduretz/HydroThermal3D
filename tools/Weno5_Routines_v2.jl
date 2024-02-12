@views function AdvectWithWeno5_v2( Tc_ex, Tc_exxx, Told, Vx, Vy, Vz, dt, _dx, _dy, _dz, Ttop, Tbot )

    @printf("Advection: WENO5\n")
    # Advection
    order = 2.0

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

    ########

    # Advect in x direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_x_Weno5!(Tc_exxx, Tc_ex, BC_type_W, BC_val_W, BC_type_E, BC_val_E)
        @parallel Advect!(Tc_ex, Tc_exxx, Vx, 1, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    # Advect in y direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_y_Weno5!(Tc_exxx, Tc_ex, BC_type_S, BC_val_S, BC_type_N, BC_val_N)
        @parallel Advect!(Tc_ex, Tc_exxx, Vy, 2, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    # Advect in z direction
    @parallel Cpy_inn_to_all!(Told, Tc_ex)
    for io=1:order
        @parallel Boundaries_z_Weno5!(Tc_exxx, Tc_ex, BC_type_B, BC_val_B, BC_type_F, BC_val_F)
        @parallel Advect!(Tc_ex, Tc_exxx, Vz, 3, dt, _dx, _dy, _dz)
    end
    @parallel TimeAveraging!(Tc_ex, Told, order)

    ####
end


############################################################

@parallel_indices (i,j,k) function Advect!(Fc::Data.Array, X, V, axis, dt::Data.Number, _dx, _dy, _dz)
    # assumes Fc is array with one extended layer
    if i<=size(Fc,1)-2 && j<=size(Fc,2)-2 && k<=size(Fc,3)-2
        I, J, K = i+3, j+3, k+3
        if axis == 1 # ---------------- X ---------------- #
            if (V[i  ,j,k] <  0.00) Vm = V[i,j,k]   end
            if (V[i  ,j,k] >= 0.00) Vm = 0.         end
            if (V[i+1,j,k] <  0.00) Vp = 0.         end
            if (V[i+1,j,k] >= 0.00) Vp = V[i+1,j,k] end 
            # ---------------- minus ------------ #
            v1    = _dx*(X[I-2, J, K] - X[I-3, J, K] )
            v2    = _dx*(X[I-1, J, K] - X[I-2, J, K] )
            v3    = _dx*(X[I-0, J, K] - X[I-1, J, K] )
            v4    = _dx*(X[I+1, J, K] - X[I-0, J, K] )
            v5    = _dx*(X[I+2, J, K] - X[I-1, J, K] )
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxm = w1*p1 + w2*p2 + w3*p3 
            # ---------------- plus ------------- #
            v1    = _dx*(X[I+3, J, K] - X[I+2, J, K] )
            v2    = _dx*(X[I+2, J, K] - X[I+1, J, K] )
            v3    = _dx*(X[I+1, J, K] - X[I+0, J, K] )
            v4    = _dx*(X[I+0, J, K] - X[I-1, J, K] )
            v5    = _dx*(X[I-1, J, K] - X[I-2, J, K] )
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxp = w1*p1 + w2*p2 + w3*p3 
        end # ---------------- Y ---------------- #
        if axis == 2
            if (V[i,j  ,k] <  0.00) Vm = V[i,j,k]   end
            if (V[i,j  ,k] >= 0.00) Vm = 0.         end
            if (V[i,j+1,k] <  0.00) Vp = 0.         end
            if (V[i,j+1,k] >= 0.00) Vp = V[i,j+1,k] end
            # ---------------- minus ------------ #
            v1    = _dy*(X[I, J-2, K] - X[I, J-3, K] )
            v2    = _dy*(X[I, J-1, K] - X[I, J-2, K] )
            v3    = _dy*(X[I, J-0, K] - X[I, J-1, K] )
            v4    = _dy*(X[I, J+1, K] - X[I, J-0, K] )
            v5    = _dy*(X[I, J+2, K] - X[I, J-1, K] )
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxm = w1*p1 + w2*p2 + w3*p3
            # ---------------- plus ------------- #
            v1    = _dy*(X[I, J+3, K] - X[I, J+2, K] )
            v2    = _dy*(X[I, J+2, K] - X[I, J+1, K] )
            v3    = _dy*(X[I, J+1, K] - X[I, J+0, K] )
            v4    = _dy*(X[I, J+0, K] - X[I, J-1, K] )
            v5    = _dy*(X[I, J-1, K] - X[I, J-2, K] ) 
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxp = w1*p1 + w2*p2 + w3*p3
        end
        if axis == 3 # ---------------- Z ---------------- #
            if (V[i,j,k  ] <  0.00) Vm = V[i,j,k]   end
            if (V[i,j,k  ] >= 0.00) Vm = 0.         end
            if (V[i,j,k+1] <  0.00) Vp = 0.         end
            if (V[i,j,k+1] >= 0.00) Vp = V[i,j,k+1] end
            # ---------------- minus ------------ #
            v1    = _dz*(X[I, J, K-2] - X[I, J, K-3] )
            v2    = _dz*(X[I, J, K-1] - X[I, J, K-2] )
            v3    = _dz*(X[I, J, K-0] - X[I, J, K-1] )
            v4    = _dz*(X[I, J, K+1] - X[I, J, K-0] )
            v5    = _dz*(X[I, J, K+2] - X[I, J, K-1] )
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxm = w1*p1 + w2*p2 + w3*p3 
            # ---------------- plus ------------- #
            v1    = _dz*(X[I, J, K+3] - X[I, J, K+2] )
            v2    = _dz*(X[I, J, K+2] - X[I, J, K+1] )
            v3    = _dz*(X[I, J, K+1] - X[I, J, K+0] )
            v4    = _dz*(X[I, J, K+0] - X[I, J, K-1] )
            v5    = _dz*(X[I, J, K-1] - X[I, J, K-2] )
            p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
            p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
            p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
            maxV  = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
            e     = 10^(-99) + 1e-6*maxV 
            w1    = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
            w2    = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
            w3    = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
            w1    = 0.1/(w1+e)^2 
            w2    = 0.6/(w2+e)^2 
            w3    = 0.3/(w3+e)^2 
            w     = (w1+w2+w3) 
            w1    = w1/w 
            w2    = w2/w 
            w3    = w3/w 
            dFdxp = w1*p1 + w2*p2 + w3*p3 
        end
        Fc[i+1,j+1,k+1] = Fc[i+1,j+1,k+1] - dt*(Vp*dFdxm + Vm*dFdxp)
    end
    return nothing
end

@parallel function  TimeAveraging!(Fc::Data.Array, Fold::Data.Array, order::Data.Number)
    # assumes Fc is array with one extended layer
    @inn(Fc) = (1.0/order)*@inn(Fc) + (1.0-1.0/order)*@all(Fold)

    return nothing
end

@parallel_indices (i,j,k) function  Boundaries_x_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_W::Int, val_W::Data.Number, type_E::Int, val_E::Data.Number)

    # assumes Fc is array with one extended layer
    if (i<=size(Fc_exxx,1)-6 && j<=size(Fc_exxx,2)-6 && k<=size(Fc_exxx,3)-6) Fc_exxx[i+3,j+3,k+3] = Fc[i+1,j+1,k+1] end

    if (type_W ==0 ) # Neumann
        if (i==1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[3+1,j-3+1,k-3+1] end
        if (i==2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[2+1,j-3+1,k-3+1] end
        if (i==3 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[1+1,j-3+1,k-3+1] end
    end

    if (type_W ==1 ) # Dirichlet
        if (i==1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_W - Fc[3+1,j-3+1,k-3+1]; end
        if (i==2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_W - Fc[2+1,j-3+1,k-3+1]; end
        if (i==3 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_W - Fc[1+1,j-3+1,k-3+1]; end
    end

    if (type_W ==2 ) # Periodic
        if (i==1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-2-1,j-3+1,k-3+1]; end
        if (i==2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-1-1,j-3+1,k-3+1]; end
        if (i==3 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-0-1,j-3+1,k-3+1]; end
    end

    if (type_E ==0 ) # Neumann
        if (i==size(Fc_exxx,1)-0 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-2-1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-1-1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[size(Fc,1)-0-1,j-3+1,k-3+1]; end
    end

    if (type_E ==1 ) # Dirichlet
        if (i==size(Fc_exxx,1)-0 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_E - Fc[size(Fc,1)-2-1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_E - Fc[size(Fc,1)-1-1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_E - Fc[size(Fc,1)-0-1,j-3+1,k-3+1]; end
    end

    if (type_E ==2 ) # Periodic
        if (i==size(Fc_exxx,1)-0 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[3+1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-1 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[2+1,j-3+1,k-3+1]; end
        if (i==size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[1+1,j-3+1,k-3+1]; end
    end
    return nothing
end

@parallel_indices (i,j,k) function  Boundaries_y_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_S::Int, val_S::Data.Number, type_N::Int, val_N::Data.Number)

    if (i<=size(Fc_exxx,1)-6 && j<=size(Fc_exxx,2)-6 && k<=size(Fc_exxx,3)-6) Fc_exxx[i+3,j+3,k+3] = Fc[i+1,j+1,k+1]; end

    if (type_S ==0 ) # Neumann
        if (i>3 && i<size(Fc_exxx,1)-2 && j==1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,3+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,2+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==3 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,1+1,k-3+1]; end
    end

    if (type_S ==1 ) # Dirichlet
        if (i>3 && i<size(Fc_exxx,1)-2 && j==1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_S - Fc[i-3+1,3+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_S - Fc[i-3+1,2+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==3 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_S - Fc[i-3+1,1+1,k-3+1]; end
    end

    if (type_S ==2 ) # Periodic
        if (i>3 && i<size(Fc_exxx,1)-2 && j==1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-2-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-1-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==3 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-0-1,k-3+1]; end
    end

    if (type_N ==0 ) # Neumann
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-0 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-2-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-1-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,size(Fc,2)-0-1,k-3+1]; end
    end

    if (type_N ==1 ) # Dirichlet
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-0 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_N - Fc[i-3+1,size(Fc,2)-2-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_N - Fc[i-3+1,size(Fc,2)-1-1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_N - Fc[i-3+1,size(Fc,2)-0-1,k-3+1]; end
    end

    if (type_N ==2 ) # Periodic
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-0 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,3+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-1 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,2+1,k-3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j==size(Fc_exxx,2)-2 && k>3 && k<size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,1+1,k-3+1]; end
    end
    return nothing
end

@parallel_indices (i,j,k) function  Boundaries_z_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_B::Int, val_B::Data.Number, type_F::Int, val_F::Data.Number)

    if (i<=size(Fc_exxx,1)-6 && j<=size(Fc_exxx,2)-6 && k<=size(Fc_exxx,3)-6) Fc_exxx[i+3,j+3,k+3] = Fc[i+1,j+1,k+1] end

    if (type_B ==0 ) # Neumann
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==1) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==2) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,2+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==3) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,1+1]; end
    end

    if (type_B ==1 ) # Dirichlet
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==1) Fc_exxx[i,j,k] = 2*val_B - Fc[i-3+1,j-3+1,3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==2) Fc_exxx[i,j,k] = 2*val_B - Fc[i-3+1,j-3+1,2+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==3) Fc_exxx[i,j,k] = 2*val_B - Fc[i-3+1,j-3+1,1+1]; end
    end

    if (type_B ==2 ) # Periodic
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==1) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-2-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==2) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-1-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==3) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-0-1]; end
    end

    if (type_F ==0 ) # Neumann
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-0) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-2-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-1) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-1-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,size(Fc,3)-0-1]; end
    end

    if (type_F ==1 ) # Dirichlet
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-0) Fc_exxx[i,j,k] = 2*val_F - Fc[i-3+1,j-3+1,size(Fc,3)-2-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-1) Fc_exxx[i,j,k] = 2*val_F - Fc[i-3+1,j-3+1,size(Fc,3)-1-1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = 2*val_F - Fc[i-3+1,j-3+1,size(Fc,3)-0-1]; end
    end

    if (type_F ==2 ) # Periodic
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-0) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,3+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-1) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,2+1]; end
        if (i>3 && i<size(Fc_exxx,1)-2 && j>3 && j<size(Fc_exxx,2)-2 && k==size(Fc_exxx,3)-2) Fc_exxx[i,j,k] = Fc[i-3+1,j-3+1,1+1]; end
    end

    return nothing
end

# @parallel function  ArrayEqualArray!(F1::Data.Array, F2::Data.Array)

#     @all(F1) = @all(F2)

#     return nothing
# end

# @parallel function  InitialCondition!(Tc::Data.Array, xc2::Data.Array, yc2::Data.Array, zc2::Data.Array, x0::Data.Number, y0::Data.Number, z0::Data.Number, sig2::Data.Number)

#     @all(Tc) = exp( -(@all(xc2)-x0)^2/sig2 - (@all(yc2)-y0)^2/sig2 - (@all(zc2)-z0)^2/sig2)

#     return nothing
# end


# @parallel_indices (i,j,k) function Advect!(Fc::Data.Array, V, dTdxm::Data.Array, dTdxp::Data.Array, axis, dt::Data.Number)
#     # assumes Fc is array with one extended layer
#     if i<=size(dTdxm,1) && j<=size(dTdxm,2) && k<=size(dTdxm,3)
#         if axis == 1
#             if (V[i  ,j,k] <  0.00) Vm = V[i,j,k]   end
#             if (V[i  ,j,k] >= 0.00) Vm = 0.         end
#             if (V[i+1,j,k] <  0.00) Vp = 0.         end
#             if (V[i+1,j,k] >= 0.00) Vp = V[i+1,j,k] end
#         end
#         if axis == 2
#             if (V[i,j  ,k] <  0.00) Vm = V[i,j,k]   end
#             if (V[i,j  ,k] >= 0.00) Vm = 0.         end
#             if (V[i,j+1,k] <  0.00) Vp = 0.         end
#             if (V[i,j+1,k] >= 0.00) Vp = V[i,j+1,k] end
#         end
#         if axis == 3
#             if (V[i,j,k  ] <  0.00) Vm = V[i,j,k]   end
#             if (V[i,j,k  ] >= 0.00) Vm = 0.         end
#             if (V[i,j,k+1] <  0.00) Vp = 0.         end
#             if (V[i,j,k+1] >= 0.00) Vp = V[i,j,k+1] end
#         end
#         Fc[i+1,j+1,k+1] = Fc[i+1,j+1,k+1] - dt*(Vp*dTdxm[i,j,k] + Vm*dTdxp[i,j,k])
#     end
#     return nothing
# end

# @parallel_indices (i,j,k) function dFdx_Weno5!(dFdxi::Data.Array, X, axis, plusminus, _dx, _dy, _dz)
# if (i<=size(dFdxi,1) && j<=size(dFdxi,2) && k<=size(dFdxi,3))
#     I, J, K = i+3, j+3, k+3
#     # ---------------- X ---------------- #
#     # ---------------- minus ------------ #
#     if axis == 1 && plusminus == -1
#         v1 = _dx*(X[I-2, J, K] - X[I-3, J, K] )
#         v2 = _dx*(X[I-1, J, K] - X[I-2, J, K] )
#         v3 = _dx*(X[I-0, J, K] - X[I-1, J, K] )
#         v4 = _dx*(X[I+1, J, K] - X[I-0, J, K] )
#         v5 = _dx*(X[I+2, J, K] - X[I-1, J, K] )
#     end
#     # ---------------- X ---------------- #
#     # ---------------- plus ------------- #
#     if axis == 1 && plusminus == 1
#         v1 = _dx*(X[I+3, J, K] - X[I+2, J, K] )
#         v2 = _dx*(X[I+2, J, K] - X[I+1, J, K] )
#         v3 = _dx*(X[I+1, J, K] - X[I+0, J, K] )
#         v4 = _dx*(X[I+0, J, K] - X[I-1, J, K] )
#         v5 = _dx*(X[I-1, J, K] - X[I-2, J, K] )
#     end
#     # ---------------- Y ---------------- #
#     # ---------------- minus ------------ #
#     if axis == 2 && plusminus == -1
#         v1 = _dy*(X[I, J-2, K] - X[I, J-3, K] )
#         v2 = _dy*(X[I, J-1, K] - X[I, J-2, K] )
#         v3 = _dy*(X[I, J-0, K] - X[I, J-1, K] )
#         v4 = _dy*(X[I, J+1, K] - X[I, J-0, K] )
#         v5 = _dy*(X[I, J+2, K] - X[I, J-1, K] )
#     end
#     # ---------------- Y ---------------- #
#     # ---------------- plus ------------- #
#     if axis == 2 && plusminus == 1
#         v1 = _dy*(X[I, J+3, K] - X[I, J+2, K] )
#         v2 = _dy*(X[I, J+2, K] - X[I, J+1, K] )
#         v3 = _dy*(X[I, J+1, K] - X[I, J+0, K] )
#         v4 = _dy*(X[I, J+0, K] - X[I, J-1, K] )
#         v5 = _dy*(X[I, J-1, K] - X[I, J-2, K] )
#     end
#     # ---------------- Z ---------------- #
#     # ---------------- minus ------------ #
#     if axis == 3 && plusminus == -1
#         v1 = _dz*(X[I, J, K-2] - X[I, J, K-3] )
#         v2 = _dz*(X[I, J, K-1] - X[I, J, K-2] )
#         v3 = _dz*(X[I, J, K-0] - X[I, J, K-1] )
#         v4 = _dz*(X[I, J, K+1] - X[I, J, K-0] )
#         v5 = _dz*(X[I, J, K+2] - X[I, J, K-1] )
#     end
#     # ---------------- Z ---------------- #
#     # ---------------- plus ------------- #
#     if axis == 3 && plusminus == 1
#         v1 = _dz*(X[I, J, K+3] - X[I, J, K+2] )
#         v2 = _dz*(X[I, J, K+2] - X[I, J, K+1] )
#         v3 = _dz*(X[I, J, K+1] - X[I, J, K+0] )
#         v4 = _dz*(X[I, J, K+0] - X[I, J, K-1] )
#         v5 = _dz*(X[I, J, K-1] - X[I, J, K-2] )
#     end
#     # Weno 5 coeffs
#     p1   = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 
#     p2   =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 
#     p3   = v3/3.0 + 5.0/6.0*v4 - v5/6.0 
#     maxV = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) 
#     e    = 10^(-99) + 1e-6*maxV 
#     w1   = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 
#     w2   = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 
#     w3   = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 
#     w1   = 0.1/(w1+e)^2 
#     w2   = 0.6/(w2+e)^2 
#     w3   = 0.3/(w3+e)^2 
#     w    = (w1+w2+w3) 
#     w1   = w1/w 
#     w2   = w2/w 
#     w3   = w3/w 
#     dFdxi[i,j,k] = w1*p1 + w2*p2 + w3*p3 
# end
# return nothing
# end