@static if USE_GPU
	macro sqrt(args...) esc(:(CUDA.sqrt($(args...)))) end
	macro exp(args...)  esc(:(CUDA.exp($(args...)))) end
else
	macro sqrt(args...) esc(:(Base.sqrt($(args...)))) end
	macro exp(args...)  esc(:(Base.exp($(args...)))) end
end

#######################

#TODO: add!
if USE_MPI
	# push!(LOAD_PATH, "./gg.jl/src") # ATTENTION: the GG package needs to be present in this subfolder - later installed normally via package manager.
	# using GG
	# import MPI
	# #TODO: these definitions are just temporarily here; they can go into the GG package. Also nprocs does not have to be get from MPI everytime if perf relevant.
	# maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))
	# minimum_g(A) = (min_l  = minimum(A); MPI.Allreduce(min_l,  MPI.MIN, MPI.COMM_WORLD))
	# mean_g(A)    = (mean_l = mean(A);    MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD)/MPI.Comm_size(MPI.COMM_WORLD))
else
	maximum_g(A) = maximum(A)
	minimum_g(A) = minimum(A)
	mean_g(A)    = mean(A)
end

#################################################
## Diverse functions
@views function diff2(A, dim)                                                          #SO: in Julia, diff does not seem to exist for 3D arrays.
	if     (dim==1); (A[3:end,:,:]-A[2:end-1,:,:]) .- (A[2:end-1,:,:]-A[1:end-2,:,:])
	elseif (dim==2); (A[:,3:end,:]-A[:,2:end-1,:]) .- (A[:,2:end-1,:]-A[:,1:end-2,:])
	elseif (dim==3); (A[:,:,3:end]-A[:,:,2:end-1]) .- (A[:,:,2:end-1]-A[:,:,1:end-2])
	end
end

# added by Thibault for the purpose of WENO-5
@views in_xxx(A)     =  A[4:end-3,4:end-3,4:end-3]

@views in_xxx_xm3(A) =  A[1:end-6,4:end-3,4:end-3]
@views in_xxx_xm2(A) =  A[2:end-5,4:end-3,4:end-3]
@views in_xxx_xm1(A) =  A[3:end-4,4:end-3,4:end-3]
@views in_xxx_xp1(A) =  A[5:end-2,4:end-3,4:end-3]
@views in_xxx_xp2(A) =  A[6:end-1,4:end-3,4:end-3]
@views in_xxx_xp3(A) =  A[7:end-0,4:end-3,4:end-3]

@views in_xxx_ym3(A) =  A[4:end-3,1:end-6,4:end-3]
@views in_xxx_ym2(A) =  A[4:end-3,2:end-5,4:end-3]
@views in_xxx_ym1(A) =  A[4:end-3,3:end-4,4:end-3]
@views in_xxx_yp1(A) =  A[4:end-3,5:end-2,4:end-3]
@views in_xxx_yp2(A) =  A[4:end-3,6:end-1,4:end-3]
@views in_xxx_yp3(A) =  A[4:end-3,7:end-0,4:end-3]

@views in_xxx_zm3(A) =  A[4:end-3,4:end-3,1:end-6]
@views in_xxx_zm2(A) =  A[4:end-3,4:end-3,2:end-5]
@views in_xxx_zm1(A) =  A[4:end-3,4:end-3,3:end-4]
@views in_xxx_zp1(A) =  A[4:end-3,4:end-3,5:end-2]
@views in_xxx_zp2(A) =  A[4:end-3,4:end-3,6:end-1]
@views in_xxx_zp3(A) =  A[4:end-3,4:end-3,7:end-0]

@views West(A)      = A[1:end-1,1:end,1:end]
@views East(A)      = A[2:end-0,1:end,1:end]
@views South(A)     = A[1:end,1:end-1,1:end]
@views North(A)     = A[1:end,2:end-0,1:end]
@views Back(A)      = A[1:end,1:end,1:end-1]
@views Front(A)     = A[1:end,1:end,2:end-0]

@views all(A)      = A[1:end,1:end,1:end]
@views inn(A)      = A[2:end-1,2:end-1,2:end-1]
@views inn_av_z(A) = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5;
@views in_x(A)     = A[2:end-1,:,:]
@views in_y(A)     = A[:,2:end-1,:]
@views in_z(A)     = A[:,:,2:end-1]
@views bc_no_dx(A) = ( A[1,:,:] .= A[2,:,:]; A[end,:,:] .= A[end-1,:,:]; ) #SO: TODO: not sure if without return nothing it will return the last statement even if it is not caught? Test later...
@views bc_no_dy(A) = ( A[:,1,:] .= A[:,2,:]; A[:,end,:] .= A[:,end-1,:]; )
@views bc_no_dz(A) = ( A[:,:,1] .= A[:,:,2]; A[:,:,end] .= A[:,:,end-1]; )
@views bc_z_val(A,val) = ( A[:,:,1] .= val; A[:,:,end] .= val; )
@views bc_dz_period(A) = ( A[:,:,1] .= A[:,:,end-1]; A[:,:,end] .= A[:,:,2]; )
detrend(A)         = A .- mean_g(A)                                               #SO: A .- mean(A) is the simplest implementation according to the MATLAB function detrend, but it should rather remove the linear trend, as I remember, right? So, it should use the derivatives, I guess.
# ## LR: save funtion
# function SaveArray(Aname, A, isave)
# 	fname = string(Aname, "_", "$(@sprintf("%05d", isave))", ".bin")
# 	out = open(fname,"w"); write(out,A); close(out)
# end

##############################
## Macros from cuda_scientific
import ParallelStencil: INDICES
ix,       iy,    iz = INDICES[1], INDICES[2], INDICES[3]
# ixi,     iyi,   izi = :($ix+1), :($iy+1), :($iz+1)
ixiii, iyiii, iziii = :($ix+3), :($iy+3), :($iz+3)
macro in_xxx(A)      esc(:( $A[$ixiii  ,$iyiii  ,$iziii   ] )) end
macro in_xxx_xm3(A)  esc(:( $A[$ixiii-3,$iyiii  ,$iziii   ] )) end
macro in_xxx_xm2(A)  esc(:( $A[$ixiii-2,$iyiii  ,$iziii   ] )) end
macro in_xxx_xm1(A)  esc(:( $A[$ixiii-1,$iyiii  ,$iziii   ] )) end
macro in_xxx_xp3(A)  esc(:( $A[$ixiii+3,$iyiii  ,$iziii   ] )) end
macro in_xxx_xp2(A)  esc(:( $A[$ixiii+2,$iyiii  ,$iziii   ] )) end
macro in_xxx_xp1(A)  esc(:( $A[$ixiii+1,$iyiii  ,$iziii   ] )) end
macro in_xxx_ym3(A)  esc(:( $A[$ixiii  ,$iyiii-3,$iziii   ] )) end
macro in_xxx_ym2(A)  esc(:( $A[$ixiii  ,$iyiii-2,$iziii   ] )) end
macro in_xxx_ym1(A)  esc(:( $A[$ixiii  ,$iyiii-1,$iziii   ] )) end
macro in_xxx_yp3(A)  esc(:( $A[$ixiii  ,$iyiii+3,$iziii   ] )) end
macro in_xxx_yp2(A)  esc(:( $A[$ixiii  ,$iyiii+2,$iziii   ] )) end
macro in_xxx_yp1(A)  esc(:( $A[$ixiii  ,$iyiii+1,$iziii   ] )) end
macro in_xxx_zm3(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii-3 ] )) end
macro in_xxx_zm2(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii-2 ] )) end
macro in_xxx_zm1(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii-1 ] )) end
macro in_xxx_zp3(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii+3 ] )) end
macro in_xxx_zp2(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii+2 ] )) end
macro in_xxx_zp1(A)  esc(:( $A[$ixiii  ,$iyiii  ,$iziii+1 ] )) end

macro  dmul_ya(A,B)  esc(:( ($A[$ix, $iy+1, $iz] * $B[$ix, $iy+1, $iz]) - ($A[$ix, $iy, $iz] * $B[$ix, $iy, $iz]) )) end

function Centroid2VerticesOnCPU!( fv, fc )
	@. fv                          = 0.0
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,1:end-1,1:end-1]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,1:end-1,1:end-1]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,2:end-0,1:end-1]
    @. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,1:end-1,2:end-0]

	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,2:end-0,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,2:end-0,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,1:end-1,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,2:end-0,1:end-1]
end

function ExCentroid2VerticesOnCPU!( fv, fc_ex )
	@. fv = 0.0
	@. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,1:end-1]
    @. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,2:end-0]

	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,1:end-1]
end

function Vertices2VyOnCPU!( fvy, fv )
	@. fvy = 0
	@. fvy += 1.0/4.0 * fv[1:end-1,:,1:end-1]
	@. fvy += 1.0/4.0 * fv[2:end-0,:,2:end-0]
	@. fvy += 1.0/4.0 * fv[1:end-1,:,2:end-0]
	@. fvy += 1.0/4.0 * fv[2:end-0,:,1:end-1]
end

function ExCentroid2VyOnCPU!( Ty, fc_ex )

	nx,ny,nz=size(fc_ex,1),size(fc_ex,2),size(fc_ex,3)
	fv  = myzeros(nx-1,ny-1,nz-1)
	fv = Array(fv);
	@. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,1:end-1]
    @. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,2:end-0]

	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,1:end-1]

	@. Ty = 0
	@. Ty += 1.0/4.0 * fv[1:end-1,:,1:end-1]
	@. Ty += 1.0/4.0 * fv[2:end-0,:,2:end-0]
	@. Ty += 1.0/4.0 * fv[1:end-1,:,2:end-0]
	@. Ty += 1.0/4.0 * fv[2:end-0,:,1:end-1]
end

######################### To become macros....

@parallel function ResetA!(A::Data.Array, B::Data.Array)

    @all(A) = 0.0
    @all(B) = 0.0

    return
end

# @parallel function Multiply!(Ty::Data.Array, ky::Data.Array, kyTy::Data.Array )

# 	@all(kyTy) = @all(ky) * @all(Ty)

# 	return
# end


@parallel function Cpy_inn_to_all!(A::Data.Array, B::Data.Array)

    @all(A) = @inn(B)

    return
end

@parallel function Cpy_all_to_inn!(A::Data.Array, B::Data.Array)

    @inn(A) = @all(B)

    return
end
