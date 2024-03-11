using Plots
plotlyjs()

kTs(T)        = 3.1138 − 0.0023*(T)
kTf(T)        = -0.069 + 0.0012*(T)
ρs(T)         = 2800*(1 - (0.000024*(T - 293)))
ρf_C(T_C,p)   = 1006+(7.424e-7*p)+(-0.3922*T_C)+(-4.441e-15*p^2)+(4.547e-9*p*T_C)+(-0.003774*T_C^2)+(1.451e-23*p^3)+(-1.793e-17*p^2*T_C)+(7.485e-12*p*T_C^2)+(2.955e-6*T_C^3)+(-1.463e-32*p^4)+(1.361e-26*p^3*T_C)+(4.018e-21*(p^2)*(T_C^2))+(-7.372e-15*p*T_C^3)+(5.698e-11*T_C^4)    
dρdP_C(T_C,p) = -7.372e-15 * T_C .^ 3 + 8.036e-21 * T_C .^ 2 .* p + 7.485e-12 * T_C .^ 2 + 4.083e-26 * T_C .* p .^ 2 - 3.586e-17 * T_C .* p + 4.547e-9 * T_C - 5.852e-32 * p .^ 3 + 4.353e-23 * p .^ 2 - 8.882e-15 * p + 7.424e-7
μf(T)         = 2.414e-5 * 10^(247.8/(T - 140.))
kF(y,δ)       = 5e-16*exp(y/δ)

mm_y          = 1000*365*3600*24

function main1D()

    # Activates non-linear solution
    nonlinear = true

    # Physical parameters 
    ymin   = -30e3
    ymax   = 0*10e3
    g      = -9.81
    ϕi     = 1e-2
    Tsurf  = 293.15
    Psurf  = 1e5
    Pmoho  = 294e6 + Psurf
    qmoho  = 37e-3
    δ      = 3000.

    # Numerical discretisation
    ncy    = 4*8-6
    Δy     = (ymax-ymin)/ncy
    yce    = LinRange(ymin-Δy/2, ymax+Δy/2, ncy+2)
    yv     = LinRange(ymin, ymax, ncy+1)

    # Arrays
    phc    = zeros(ncy+2)
    T      = zeros(ncy+2)
    P      = zeros(ncy+2)
    RT     = zeros(ncy+0)
    RH     = zeros(ncy+0)
    qT     = zeros(ncy+1)
    qH     = zeros(ncy+1)
    kT     = zeros(ncy+1)
    kH     = zeros(ncy+1)
    kH_D   = zeros(ncy+1)
    Tv     = zeros(ncy+1)
    Pv     = zeros(ncy+1)
    ρ      = zeros(ncy+1)
    Vy     = zeros(ncy+1)

    # Treatment of immersed surface
    ysurf = 0e3
    tiny  = 1e-9
    phc[yce.>ysurf-tiny] .= 1
    phc[yce.>ysurf-tiny .&& yce.<(ysurf-tiny + 3*Δy/2)] .= -1
    ind   = findfirst(phc.==-1.0)
    wN    = 1-(yce[ind]-ysurf)/Δy # Interface weight

    # Initial thermal condition, prior to solving for a static non-linear solution
    dTdy   = -20e-3
    T     .= yce*dTdy .+ Tsurf
    Tv    .= 0.5.*(T[1:end-1] .+ T[2:end])
    kT    .= ϕi.*kTf.(Tv) .+ (1 .- ϕi).*kTs.(Tv)

    # Initial hydraulic condition, prior to solving for a static non-linear solution
    dPdy   = (Psurf-Pmoho)/30e3
    P     .= yce*dPdy .+ Psurf
    P[phc.==1] .= Psurf
    Pv    .= 0.5.*(P[1:end-1] .+ P[2:end])
    kH    .= ρf_C.(Tv .- 273.15, Pv) .* kF.(yv, δ) ./  μf.(Tv)

    # Pseudo-time integration steps
    errT0, errP0 = 1.0, 1.0
    ΔτT   = Δy^2/(maximum(kT))/2.1 / 3
    ΔτH   = Δy^2/(maximum(kH))/2.1 # * 2e18 

    # Solution procedure
    for it=1:10000000

        for i in eachindex(T)
            if i==1 # MOHO
                # Thermal: Fixed flux
                T[1]   = qmoho*Δy/kT[1] + T[2]
                # Hydro: Fixed pressure
                # P[1]   = 2*Pmoho - P[2]
                P[1]   =  P[2]
            elseif phc[i]==-1 # SURFACE
                # Thermal: Fixed temperature
                T[i] = (Tsurf - (1-wN)*T[i-1])/wN    
                #  Hydro: Fixed pressure
                P[i] = (Psurf - (1-wN)*P[i-1])/wN  
            elseif phc[i]==1 # Above empty cells (for finite difference code - irrelevant for finite element solution)
                T[i] = Tsurf
                P[i] = Psurf
            end
        end

        # THERMAL PROBLEM
        if nonlinear
            Tv    .= 0.5.*(T[1:end-1] .+ T[2:end])
            kT    .= ϕi.*kTf.(Tv) .+ (1 .- ϕi).*kTs.(Tv) 
        end
        qT    .= .-kT.* (T[2:end] .- T[1:end-1])/Δy
        RT    .= .-(qT[2:end] .- qT[1:end-1])/Δy
        RT[phc[2:end-1].==1 .|| phc[2:end-1].==-1] .= 0.
        T[2:end-1] .+= RT.*ΔτT

        # HYDRAULIC PROBLEM
        if nonlinear
            Pv    .= 0.5.*(P[1:end-1] .+ P[2:end])
            kH    .= ρf_C.(Tv .- 273.15, Pv) .* kF.(yv, δ) ./  μf.(Tv)
        end
        qH    .= .-kH.* ( (P[2:end] .- P[1:end-1])/Δy - ρf_C.(Tv .- 273.15, Pv)*g)
        RH    .= .-(qH[2:end] .- qH[1:end-1])/Δy
        RH[phc[2:end-1].==1 .|| phc[2:end-1].==-1] .= 0.
        P[2:end-1] .+= RH.*ΔτH # * 2e18

        if mod(it, 500)==0
            errT, errP = norm(RT), norm(RH)
            if it==1 errT0 = errT; errP0 = errP end 
            @show errT/errT0, errP/errP0
            if errP/errP0<1e-22 && errT/errT0<1e-16 break end
            if isnan(errT/errT0) break end
        end

    end


    # Exact Moho values by interpolation
    Tmoho = 0.5*(T[1] + T[2])
    Pmoho = 0.5*(P[1] + P[2])

    # Fluid density and Darcy velocity
    ρ    .= ρf_C.(Tv .- 273.15, Pv)
    kH_D .= kF.(yv, δ) ./  μf.(Tv)
    Vy   .=  .-kH_D.* ( (P[2:end] .- P[1:end-1])/Δy - ρ*g)

    # Strings for figure titles
    Tmoho_str  = @sprintf("%1.3lf", Tmoho-273.15)
    qTmoho_str = @sprintf("%1.3lf", qT[1])
    kTmoho_str = @sprintf("%1.3lf", kT[1])
    Pmoho_str  = @sprintf("%1.3lf", Pmoho/1e6)
    ρmoho_str  = @sprintf("%1.3lf", ρ[1])
    Vmoho_str  = @sprintf("%1.3lf", Vy[1]*mm_y)
    # qHmoho_str = @sprintf("%1.3e",  qH[1])
    # kHmoho_str = @sprintf("%1.3e",  kH[1])

    # Plots
    p1 = plot( T.-273, yce/1e3, title="T  moho = $Tmoho_str C", label=:none, marker=:dot, titlefont = font(12,"Computer Modern"))
    p2 = plot(     qT,  yv/1e3, title="qT moho = $qTmoho_str W/m2", label=:none, marker=:dot, xlim=(30e-3,40e-3), titlefont = font(12,"Computer Modern"))
    p3 = plot(     kT,  yv/1e3, title="kT moho = $kTmoho_str W/m/K", label=:none, marker=:dot, titlefont = font(12,"Computer Modern"))

    p4 = plot( P./1e6,  yce/1e3, title="P moho = $Pmoho_str MPa", label=:none, marker=:dot, titlefont = font(12,"Computer Modern"))
    p5 = plot( Vy.*mm_y, yv/1e3, title="V moho = $Vmoho_str mm/y", label=:none, marker=:dot, titlefont = font(12,"Computer Modern"))
    p6 = plot(        ρ, yv/1e3, title="ρ moho = $ρmoho_str kg/m3", label=:none, marker=:dot, titlefont = font(12,"Computer Modern"))

    # p5 = plot(     qH,  yv/1e3, title="qH moho = $qHmoho_str Pa/m/s", label=:none, xlim=(-1e-4,1e-4), titlefont = font(12,"Computer Modern"))
    # p6 = plot(     kH,  yv/1e3, title="kH moho = $kHmoho_str s", label=:none, titlefont = font(12,"Computer Modern"))

    display(plot(p1,p2,p3, p4,p5,p6))
    
end

main1D()