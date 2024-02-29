using Plots
gr()

function main()

    Ra       = 1600.0 
    dT       = 1.0
    Ttop     = 0.0
    Tbot     = Ttop + dT
    dP       = 1.0
    Ptop     = 0.0
    Pbot     = Ptop + dP
    rho_cp   = 1.0
    k        = 1.0
    lam0     = 1.0
    time     = 0
    β        = 1e-3
    xmin     = -0.5;  xmax =  0.5; Lx = xmax - xmin
    ymin     = -0.5;  ymax =  0.5; Ly = ymax - ymin
    # Numerics
    nt       = 1
    nout     = 1
    ncx      = 2*32-6
    ncy      = 2*32-6
    dx, dy   = Lx/ncx, Ly/ncy  

    Vx       = zeros(ncx+1, ncy)
    Vy       = zeros(ncx, ncy+1)
    qx       = zeros(ncx+1, ncy)
    qy       = zeros(ncx, ncy+1)
    T        = zeros(ncx+2, ncy+2)
    P        = zeros(ncx+2, ncy+2)
    Rp       = zeros(ncx+0, ncy+0)
    time     = 0.

    xc  = LinRange(xmin+dx/2, xmax-dx/2, ncx+0)
    yc  = LinRange(ymin+dy/2, ymax-dy/2, ncy+0)
    xv  = LinRange(xmin, xmax, ncx+1)
    yv  = LinRange(ymin, ymax, ncy+1)
    xce = LinRange(xmin-dx/2, xmax+dx/2, ncx+2)
    yce = LinRange(ymin-dy/2, ymax+dy/2, ncy+2)
    dTdy = -1.0
    dPdy = -1.0
    @. T   .= xce*0. .+ (yce.-ymax)'.*dTdy
    @. P   .= xce*0. .+ (yce.-ymax)'.*dPdy
    @. T  .+= exp.(.-(xce.^2 .+ ((yce+ymax/2).^2)') .*200)

    for it = 1:nt

        # T[  1,:] .= T[    2,:]
        # T[end,:] .= T[end-1,:]
        # T[:,  1] .= 2*Tbot .- T[:,    2]
        # T[:,end] .= 2*Ttop .- T[:,end-1]

        # P[  1,:] .= P[    2,:]
        # P[end,:] .= P[end-1,:]
        # P[:,  1] .= P[:,    2]
        # P[:,end] .= P[:,end-1]

        # @. Vx .= -k*(  ( P[2:end-0,2:end-1] - P[1:end-1,2:end-1])  )/dx
        # @. Vy .= -k*(  ( P[2:end-1,2:end-0] - P[2:end-1,1:end-1])  )/dy + Ra*(T[2:end-1,1:end-1] + T[2:end-1,2:end-0])/2
        # Vy[:,1]   .= 0.
        # Vy[:,end] .= 0.

        # dt_adv   = 0.24999* min(dx, dy)/max(maximum(abs.(Vx)), maximum(abs.(Vy)))
        # dt_diffT = 0.24999* min(dx, dy)^2 /(lam0/rho_cp)
        # dt_diffP = 0.24999* min(dx, dy)^2 /(k/β)
        # dt       = min(dt_adv, dt_diffT, dt_diffP)

        # @. qx = Vx 
        # @. qy = Vy 
        # @. Rp = -( (qx[2:end-0,:] - qx[1:end-1,:])/dx + (qy[:,2:end-0] - qy[:,1:end-1])/dy )
        # @. P[2:end-1,2:end-1] -= dt/β *( (qx[2:end-0,:] - qx[1:end-1,:])/dx + (qy[:,2:end-0] - qy[:,1:end-1])/dy )

        # # Diffusion
        # @. qx = - lam0*(T[2:end-0,2:end-1] - T[1:end-1,2:end-1])/dx
        # @. qy = - lam0*(T[2:end-1,2:end-0] - T[2:end-1,1:end-1])/dy
        # @. T[2:end-1,2:end-1] -= dt/rho_cp * ( (qx[2:end-0,:] - qx[1:end-1,:])/dx + (qy[:,2:end-0] - qy[:,1:end-1])/dy )

        # # Advection
        # @. qx = (Vx<0.0)*T[2:end-0,2:end-1].*Vx + (Vx>=0.0)*T[1:end-1,2:end-1].*Vx
        # @. qy = (Vy<0.0)*T[2:end-1,2:end-0].*Vy + (Vy>=0.0)*T[2:end-1,1:end-1].*Vy
        # @. T[2:end-1,2:end-1] -= dt* ( (qx[2:end-0,:] - qx[1:end-1,:])/dx + (qy[:,2:end-0] - qy[:,1:end-1])/dy )

        # time += dt

        if mod(it, nout)==0 || it==1
            # p = heatmap(xce, yce, T', title=time, aspect_ratio=1, xlims=(-0.5,0.5)) 
            # p = heatmap(xce, yce, P', aspect_ratio=1, xlims=(-0.5,0.5), title="Initial pressure", xlabel="x []", ylabel="y []") 
            p = heatmap(xce, yce, T', aspect_ratio=1, xlims=(-0.5,0.5), title="Initial temperature", xlabel="x []", ylabel="y []") 

            # p = heatmap(xc, yv, Vy') 
            display(p)
            sleep(0.2)
        end
    end
end

main()
    
#   dt       = min(dt_adv, dt_diffT)
# # Diffusion
# for iter=1:1
#     P[  1,:] .= P[    2,:]
#     P[end,:] .= P[end-1,:]
#     P[:,  1] .= P[:,    2]
#     P[:,end] .= P[:,end-1]
#     @. Vx .= -k*(  ( P[2:end-0,2:end-1] - P[1:end-1,2:end-1])  )/dx
#     @. Vy .= -k*(  ( P[2:end-1,2:end-0] - P[2:end-1,1:end-1])  )/dy - g*Ra*(T[2:end-1,1:end-1] + T[2:end-1,2:end-0])/2
#     Vy[:,1]   .= 0.
#     Vy[:,end] .= 0.
    # @. qx = Vx 
    # @. qy = Vy 
    # @. Rp = -( (qx[2:end-0,:] - qx[1:end-1,:])/dx + (qy[:,2:end-0] - qy[:,1:end-1])/dy )
#     @show norm(Rp)
#     @. P[2:end-1,2:end-1] += dt_diffP * Rp /β
# end
# @show norm(Rp)`