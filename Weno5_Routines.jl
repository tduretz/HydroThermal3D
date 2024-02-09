############################################################
# DatArray_k --> Data.Array
# DAT --> Data.Number

@parallel_indices (ix,iy,iz) function VxPlusMinus!(Vxm::Data.Array, Vxp::Data.Array, Vx::Data.Array)

    if (ix<=size(Vxm,1) && iy<=size(Vxm,2) && iz<=size(Vxm,3))
        if (Vx[ix,iy,iz] < 0.00) Vxm[ix,iy,iz] = 0.00         end
        if (Vx[ix,iy,iz] > 0.00) Vxm[ix,iy,iz] = Vx[ix,iy,iz] end
    end
    if (ix<=size(Vxp,1) && iy<=size(Vxp,2) && iz<=size(Vxp,3))
        if (Vx[ix+1,iy,iz] < 0.00) Vxp[ix,iy,iz] = Vx[ix+1,iy,iz] end
        if (Vx[ix+1,iy,iz] > 0.00) Vxp[ix,iy,iz] = 0.00           end
    end
    return
end

@parallel_indices (ix,iy,iz) function VyPlusMinus!(Vym::Data.Array, Vyp::Data.Array, Vy::Data.Array)

    if (ix<=size(Vym,1) && iy<=size(Vym,2) && iz<=size(Vym,3))
        if (Vy[ix,iy,iz] > 0.00) Vym[ix,iy,iz] = Vy[ix,iy,iz] end
        if (Vy[ix,iy,iz] < 0.00) Vym[ix,iy,iz] = 0.00         end
    end
    if (ix<=size(Vyp,1) && iy<=size(Vyp,2) && iz<=size(Vyp,3))
        if (Vy[ix,iy+1,iz] > 0.00) Vyp[ix,iy,iz] = 0.00           end
        if (Vy[ix,iy+1,iz] < 0.00) Vyp[ix,iy,iz] = Vy[ix,iy+1,iz] end
    end
    return
 end

@parallel_indices (ix,iy,iz) function VzPlusMinus!(Vzm::Data.Array, Vzp::Data.Array, Vz::Data.Array)

    if (ix<=size(Vzm,1) && iy<=size(Vzm,2) && iz<=size(Vzm,3))
        if (Vz[ix,iy,iz] > 0.00) Vzm[ix,iy,iz] = Vz[ix,iy,iz] end
        if (Vz[ix,iy,iz] < 0.00) Vzm[ix,iy,iz] = 0.00         end
    end
    if (ix<=size(Vzp,1) && iy<=size(Vzp,2) && iz<=size(Vzp,3))
        if (Vz[ix,iy,iz+1] > 0.00) Vzp[ix,iy,iz] = 0.00           end
        if (Vz[ix,iy,iz+1] < 0.00) Vzp[ix,iy,iz] = Vz[ix,iy,iz+1] end
    end
    return
end

 @parallel_indices (ix,iy,iz) function dFdx_Weno5!(dFdxi::Data.Array, V1::Data.Array, V2::Data.Array, V3::Data.Array, V4::Data.Array, V5::Data.Array)

    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) v1   = V1[ix,iy,iz] end
    if (ix<=size(V2,1) && iy<=size(V2,2) && iz<=size(V2,3)) v2   = V2[ix,iy,iz] end
    if (ix<=size(V3,1) && iy<=size(V3,2) && iz<=size(V3,3)) v3   = V3[ix,iy,iz] end
    if (ix<=size(V4,1) && iy<=size(V4,2) && iz<=size(V4,3)) v4   = V4[ix,iy,iz] end
    if (ix<=size(V5,1) && iy<=size(V5,2) && iz<=size(V5,3)) v5   = V5[ix,iy,iz] end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) p1   = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) p2   =-v2/6.0 + 5.0/6.0*v3 + v4/3.0 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) p3   = v3/3.0 + 5.0/6.0*v4 - v5/6.0 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) maxV = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2) end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) e    = 10^(-99) + 1e-6*maxV end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w1   = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w2   = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w3   = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w1   = 0.1/(w1+e)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w2   = 0.6/(w2+e)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w3   = 0.3/(w3+e)^2 end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w    = (w1+w2+w3) end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w1   = w1/w end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w2   = w2/w end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) w3   = w3/w end
    if (ix<=size(V1,1) && iy<=size(V1,2) && iz<=size(V1,3)) dFdxi[ix,iy,iz] = w1*p1 + w2*p2 + w3*p3 end
    return
end

@parallel function Gradients_minus_x_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dx*( @in_xxx_xm2(Fc_exxx) -  @in_xxx_xm3(Fc_exxx) )
    @all(v2) = _dx*( @in_xxx_xm1(Fc_exxx) -  @in_xxx_xm2(Fc_exxx) )
    @all(v3) = _dx*(     @in_xxx(Fc_exxx) -  @in_xxx_xm1(Fc_exxx) )
    @all(v4) = _dx*( @in_xxx_xp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v5) = _dx*( @in_xxx_xp2(Fc_exxx) -  @in_xxx_xp1(Fc_exxx) )
    return
end

@parallel function Gradients_plus_x_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dx*( @in_xxx_xp3(Fc_exxx) -  @in_xxx_xp2(Fc_exxx) )
    @all(v2) = _dx*( @in_xxx_xp2(Fc_exxx) -  @in_xxx_xp1(Fc_exxx) )
    @all(v3) = _dx*( @in_xxx_xp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v4) = _dx*(     @in_xxx(Fc_exxx) -  @in_xxx_xm1(Fc_exxx) )
    @all(v5) = _dx*( @in_xxx_xm1(Fc_exxx) -  @in_xxx_xm2(Fc_exxx) )
    return
end

@parallel function Gradients_minus_y_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dy*( @in_xxx_ym2(Fc_exxx) -  @in_xxx_ym3(Fc_exxx) )
    @all(v2) = _dy*( @in_xxx_ym1(Fc_exxx) -  @in_xxx_ym2(Fc_exxx) )
    @all(v3) = _dy*(     @in_xxx(Fc_exxx) -  @in_xxx_ym1(Fc_exxx) )
    @all(v4) = _dy*( @in_xxx_yp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v5) = _dy*( @in_xxx_yp2(Fc_exxx) -  @in_xxx_yp1(Fc_exxx) )
    return
end

@parallel function Gradients_plus_y_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dy*( @in_xxx_yp3(Fc_exxx) -  @in_xxx_yp2(Fc_exxx) )
    @all(v2) = _dy*( @in_xxx_yp2(Fc_exxx) -  @in_xxx_yp1(Fc_exxx) )
    @all(v3) = _dy*( @in_xxx_yp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v4) = _dy*(     @in_xxx(Fc_exxx) -  @in_xxx_ym1(Fc_exxx) )
    @all(v5) = _dy*( @in_xxx_ym1(Fc_exxx) -  @in_xxx_ym2(Fc_exxx) )
    return
end

@parallel function Gradients_minus_z_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dz*( @in_xxx_zm2(Fc_exxx) -  @in_xxx_zm3(Fc_exxx) )
    @all(v2) = _dz*( @in_xxx_zm1(Fc_exxx) -  @in_xxx_zm2(Fc_exxx) )
    @all(v3) = _dz*(     @in_xxx(Fc_exxx) -  @in_xxx_zm1(Fc_exxx) )
    @all(v4) = _dz*( @in_xxx_zp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v5) = _dz*( @in_xxx_zp2(Fc_exxx) -  @in_xxx_zp1(Fc_exxx) )
    return
end

@parallel function Gradients_plus_z_Weno5!(v1::Data.Array, v2::Data.Array, v3::Data.Array, v4::Data.Array, v5::Data.Array, Fc_exxx::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)

    @all(v1) = _dz*( @in_xxx_zp3(Fc_exxx) -  @in_xxx_zp2(Fc_exxx) )
    @all(v2) = _dz*( @in_xxx_zp2(Fc_exxx) -  @in_xxx_zp1(Fc_exxx) )
    @all(v3) = _dz*( @in_xxx_zp1(Fc_exxx) -      @in_xxx(Fc_exxx) )
    @all(v4) = _dz*(     @in_xxx(Fc_exxx) -  @in_xxx_zm1(Fc_exxx) )
    @all(v5) = _dz*( @in_xxx_zm1(Fc_exxx) -  @in_xxx_zm2(Fc_exxx) )
    return
end

@parallel_indices (ix,iy,iz) function  Boundaries_x_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_W::Int, val_W::Data.Number, type_E::Int, val_E::Data.Number)

    if (ix<=size(Fc_exxx,1)-6 && iy<=size(Fc_exxx,2)-6 && iz<=size(Fc_exxx,3)-6) Fc_exxx[ix+3,iy+3,iz+3] = Fc[ix,iy,iz] end

    if (type_W ==0 ) # Neumann
        if (ix==1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[3,iy-3,iz-3] end
        if (ix==2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[2,iy-3,iz-3] end
        if (ix==3 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[1,iy-3,iz-3] end
    end

    if (type_W ==1 ) # Dirichlet
        if (ix==1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[3,iy-3,iz-3]; end
        if (ix==2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[2,iy-3,iz-3]; end
        if (ix==3 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[1,iy-3,iz-3]; end
    end

    if (type_W ==2 ) # Periodic
        if (ix==1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-2,iy-3,iz-3]; end
        if (ix==2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-1,iy-3,iz-3]; end
        if (ix==3 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-0,iy-3,iz-3]; end
    end

    if (type_E ==0 ) # Neumann
        if (ix==size(Fc_exxx,1)-0 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-2,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-1,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[size(Fc,1)-0,iy-3,iz-3]; end
    end

    if (type_E ==1 ) # Dirichlet
        if (ix==size(Fc_exxx,1)-0 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[size(Fc,1)-2,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[size(Fc,1)-1,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[size(Fc,1)-0,iy-3,iz-3]; end
    end

    if (type_E ==2 ) # Periodic
        if (ix==size(Fc_exxx,1)-0 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[3,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-1 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[2,iy-3,iz-3]; end
        if (ix==size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[1,iy-3,iz-3]; end
    end
    return
end

@parallel_indices (ix,iy,iz) function  Boundaries_y_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_S::Int, val_S::Data.Number, type_N::Int, val_N::Data.Number)

    if (ix<=size(Fc_exxx,1)-6 && iy<=size(Fc_exxx,2)-6 && iz<=size(Fc_exxx,3)-6) Fc_exxx[ix+3,iy+3,iz+3] = Fc[ix,iy,iz]; end

    if (type_S ==0 ) # Neumann
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,3,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==3 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,1,iz-3]; end
    end

    if (type_S ==1 ) # Dirichlet
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,3,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==3 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,1,iz-3]; end
    end

    if (type_S ==2 ) # Periodic
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-1,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==3 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-0,iz-3]; end
    end

    if (type_N ==0 ) # Neumann
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-0 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-1,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,size(Fc,2)-0,iz-3]; end
    end

    if (type_N ==1 ) # Dirichlet
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-0 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,size(Fc,2)-2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,size(Fc,2)-1,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,size(Fc,2)-0,iz-3]; end
    end

    if (type_N ==2 ) # Periodic
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-0 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,3,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-1 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,2,iz-3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy==size(Fc_exxx,2)-2 && iz>3 && iz<size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,1,iz-3]; end
    end
    return
end

@parallel_indices (ix,iy,iz) function  Boundaries_z_Weno5!(Fc_exxx::Data.Array, Fc::Data.Array, type_B::Int, val_B::Data.Number, type_F::Int, val_F::Data.Number)

    if (ix<=size(Fc_exxx,1)-6 && iy<=size(Fc_exxx,2)-6 && iz<=size(Fc_exxx,3)-6) Fc_exxx[ix+3,iy+3,iz+3] = Fc[ix,iy,iz] end

    if (type_B ==0 ) # Neumann
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==3) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,1]; end
    end

    if (type_B ==1 ) # Dirichlet
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==1) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==2) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==3) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,1]; end
    end

    if (type_B ==2 ) # Periodic
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-1]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==3) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-0]; end
    end

    if (type_F ==0 ) # Neumann
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-0) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-1]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,size(Fc,3)-0]; end
    end

    if (type_F ==1 ) # Dirichlet
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-0) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,size(Fc,3)-2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-1) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,size(Fc,3)-1]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,size(Fc,3)-0]; end
    end

    if (type_F ==2 ) # Periodic
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-0) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,3]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,2]; end
        if (ix>3 && ix<size(Fc_exxx,1)-2 && iy>3 && iy<size(Fc_exxx,2)-2 && iz==size(Fc_exxx,3)-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,1]; end
    end

    return
end

@parallel function  Advect!(Fc::Data.Array, Vxp::Data.Array, dTdxm::Data.Array, Vxm::Data.Array, dTdxp::Data.Array, dt::Data.Number)

    @all(Fc) = @all(Fc) - dt*(@all(Vxp)*@all(dTdxm) + @all(Vxm)*@all(dTdxp))

    return
end

@parallel function  TimeAveraging!(Fc::Data.Array, Fold::Data.Array, order::Data.Number)

    @all(Fc) = (1.0/order)*@all(Fc) + (1.0-1.0/order)*@all(Fold)

    return
end

@parallel function  ArrayEqualArray!(F1::Data.Array, F2::Data.Array)

    @all(F1) = @all(F2)

    return
end

@parallel function  InitialCondition!(Tc::Data.Array, xc2::Data.Array, yc2::Data.Array, zc2::Data.Array, x0::Data.Number, y0::Data.Number, z0::Data.Number, sig2::Data.Number)

    @all(Tc) = exp( -(@all(xc2)-x0)^2/sig2 - (@all(yc2)-y0)^2/sig2 - (@all(zc2)-z0)^2/sig2)

    return
end
