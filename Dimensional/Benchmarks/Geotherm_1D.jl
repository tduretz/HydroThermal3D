using Plots

kTs(T)        = 3.1138 − 0.0023*(T)
kTf(T)        = -0.069 + 0.0012*(T)

function main1D()

    ϕi     = 1e-2
    Tsurf  = 293.15
    qmoho  = 37e-3

    ymin   = -30e3
    ymax   = 1*10e3
    ncy    = 4*8-6
    @show Δy     = (ymax-ymin)/ncy
    yce    = LinRange(ymin-Δy/2, ymax+Δy/2, ncy+2)
    yv     = LinRange(ymin, ymax, ncy+1)

    phc    = zeros(ncy+2)
    T      = zeros(ncy+2)
    R      = zeros(ncy+0)
    q      = zeros(ncy+1)
    k      = zeros(ncy+1)
    Tv     = zeros(ncy+1)

    # Condition initiale avant résolution du probleme stationaire non-lineaire
    ysurf = 0e3
    tiny  = 1e-9
    phc[yce.>ysurf-tiny] .= 1
    phc[yce.>ysurf-tiny .&& yce.<(ysurf-tiny + 3*Δy/2)] .= -1
    ind = findfirst(phc.==-1.0)
    @show wN     = 1-(yce[ind]-ysurf)/Δy
    # if abs(wN)<1e-13 error("surface node coincide with interface: move surface or change box height") end
    T     .= -yce*20e-3 .+ Tsurf
    Tv    .= 0.5.*(T[1:end-1] .+ T[2:end])
    k     .= ϕi.*kTf.(Tv) .+ (1 .- ϕi).*kTs.(Tv)

    # Resolution du probleme stationaire non-lineaire
    err0 = 1.0
    Δτ   = Δy^2/(maximum(k))/2.1

    for it=1:5000000

        for i in eachindex(T)
            # if i==1
            #     T[1]   = qmoho*Δy/k[1] + T[2]
            # elseif i==ncy+2
            #     T[i] = 2*Tsurf - T[i-1]
            # end
            if i==1
                T[1]   = qmoho*Δy/k[1] + T[2]
            elseif phc[i]==-1 
                T[i] = (Tsurf - (1-wN)*T[i-1])/wN    # Tbc = wN*TN + (1-wN)*TS = wN*TN - wN*TS + TS
            elseif phc[i]==1 
                T[i] = Tsurf
            end
        end
        # T[end] = Tsurf # wrong BC converges badly

        # Tv    .= 0.5.*(T[1:end-1] .+ T[2:end])
        # k     .= ϕi.*kTf.(Tv) .+ (1 .- ϕi).*kTs.(Tv)
        q     .= .-k .* (T[2:end] .- T[1:end-1])/Δy
        R     .= .-(q[2:end] .- q[1:end-1])/Δy
        R[phc[2:end-1].==1 .|| phc[2:end-1].==-1] .= 0.
        T[2:end-1] .+= R.*Δτ/3

        if mod(it, 50000)==0
            err = norm(R)
            if it==1 err0=err end 
            @show err/err0
            if err/err0<1e-15 break end
            if isnan(err/err0) break end
        end
    end

    # Tv    .= 0.5.*(T[1:end-1] .+ T[2:end])
    # k     .= ϕi.*kTf.(Tv) .+ (1 .- ϕi).*kTs.(Tv)
    # q     .= .-k .* (T[2:end] .- T[1:end-1])/Δy


    Tmoho = 0.5*(T[1]+T[2])
    p1 = plot( T.-273, yce/1e3, title="T moho = $(Tmoho-273.15)", label=:none, marker=:dot)
    p2 = plot(      q,  yv/1e3, title="q moho = $(q[1])", label=:none, xlim=(30e-3,40e-3))
    p3 = plot(      k,  yv/1e3, title="k moho = $(k[1])", label=:none)
    p3 = plot(     phc,  yce/1e3, title="k moho = $(k[1])", label=:none)

    display(plot(p1,p2,p3))
    
end

main1D()