# https://users.wpi.edu/~walker/MA512/HANDOUTS/MATLAB_ODE_Suite.pdf
using DocStringExtensions
using LinearAlgebra
using StaticArrays
using Parameters
using ForwardDiff
using DiffEqBase
using Printf

debugmode() = false

struct QNDFOptions{AT,RT,I,SB,SA,QA,QI}
    abstol::AT
    reltol::RT
    internalnorm::I
    newton_max_iters::Int
    step_bias::SB
    step_addon::SA
    qmax::QA
    qmin::QI
end

Base.@kwdef mutable struct QNDFState{O,BDF,F,paramType,T,uType,rateType,N,JacType,WType,gtType,bType,gbType,UType,gType,eType,pType,RUType,OType,EEstType}
    maxorder::Val{O}; isbdf::Val{BDF}
    f::F; p::paramType
    orderprev::Int; order::Int
    n_constant_steps::Int; isfinal::Bool; EEst::EEstType
    tdir::T; t::T; tn::T; tend::T; dtprev::T; dt::T; dtmin::T; dtmax::T
    uprev::uType; u::uType
    predictor::uType; Ψ::pType; ∇p1::rateType; ∇::N; ∇tmp::N
    J::JacType; W::WType; γdt::gtType
    𝔹::bType; γ𝔹::gbType; U::UType; γs::gType; εs::eType; RU::RUType
    opts::OType; destats::DiffEqBase.DEStats
end

function init(
        prob, dt=1e-6, maxorder::Val{MAX_ORDER}=Val(5), isbdf::Val{BDF}=Val(false);
        abstol=1e-3, reltol=1e-6,
        internalnorm=DiffEqBase.ODE_DEFAULT_NORM,
        newton_max_iters=10,
        # Step size parameters from LSODA:
        # order+1: https://github.com/lh3/misc/blob/cc0f36a9a19f35765efb9387389d9f3a6756f08f/math/lsoda.c#L1791-L1792
        # order: https://github.com/lh3/misc/blob/cc0f36a9a19f35765efb9387389d9f3a6756f08f/math/lsoda.c#L2640-L2641
        # order-1: https://github.com/lh3/misc/blob/cc0f36a9a19f35765efb9387389d9f3a6756f08f/math/lsoda.c#L2646-L2647
        step_bias=(1.3, 1.2, 1.4), # order-1, order, order+1
        step_addon=step_bias .* 1e-6, # order-1, order, order+1
        qmax=10, qmin=0.1,
    ) where {MAX_ORDER, BDF}
    opts = QNDFOptions(
        abstol, reltol, internalnorm,
        newton_max_iters,
        step_bias, step_addon, qmax, qmin,
       )

    @unpack f, u0, p, tspan = prob
    @unpack mass_matrix = f
    tup = MAX_ORDER-4 >= 1 ? ntuple(_->0.0, MAX_ORDER-4) : ()
    κs = BDF ? ntuple(_->false, MAX_ORDER) : (-0.185, -1/9, -0.0823, -0.0415, tup...)

    𝔹 = SVector(ntuple(n->sum(1//j for j in 1:n), Val(MAX_ORDER))) # BDF coefficients
    U = SMatrix{MAX_ORDER,MAX_ORDER}([
                                      Int(prod(m->m-r, 0:j-1)//factorial(j))
                                      for j in 1:MAX_ORDER, r in 1:MAX_ORDER]
                                    ) |> UpperTriangular
    γs = ntuple(Val(MAX_ORDER)) do k
        b = 𝔹[k]; κ = κs[k]
        γ = inv((1 - κ) * b) # Section 4
    end |> SVector
    εs = ntuple(Val(MAX_ORDER)) do k
        b = 𝔹[k]; κ = κs[k]
        ε = κ * b + 1//(k+1) # (5)
    end |> SVector
    γ𝔹 = Vector(γs[1] * 𝔹)

    t, tend = tspan
    tdir = sign(tend - t)
    tn = t
    dtprev = dt

    uprev = u0
    u = copy(uprev)
    predictor = zero(vec(u))
    ∇p1 = zero(vec(u))
    destats = DiffEqBase.DEStats(0)
    fu = f(u, p, t); destats.nf += 1
    ∇ = similar(fu, Base.OneTo(length(u0)), Base.OneTo(MAX_ORDER+2))
    @views @. ∇[:, 1] = fu * dt
    @views fill!(∇[:, 2:end], zero(eltype(∇)))
    ∇tmp = similar(∇)
    @views fill!(∇tmp, zero(eltype(∇)))
    RU = similar(∇, MAX_ORDER, MAX_ORDER)
    Ψ = similar(∇, Base.OneTo(length(u0)))

    γ, ε = first(γs), first(εs)
    γdt = γ * dt
    J = jacobian(f, uprev, p, t); destats.njacs += 1
    W = lu!(mass_matrix - γdt * J); destats.nw += 1

    QNDFState(;
              maxorder=maxorder, isbdf=isbdf, f=prob.f, p=prob.p,
              orderprev=1, order=1, tdir=tdir, t=t, tn=tn, tend=tend,
              dt=dt, dtprev=dtprev, dtmin=dt, dtmax=dt,
              n_constant_steps=0,
              uprev=uprev, u=u, predictor=predictor,
              ∇=∇, ∇tmp=∇tmp, ∇p1=∇p1,
              𝔹=𝔹, γ𝔹=γ𝔹, U=U, γs=γs, εs=εs, Ψ=Ψ, RU=RU,
              J=J, W=W, γdt=γdt,
              opts=opts, destats=destats,
              isfinal=false, EEst=one(eltype(fu)),
             )
end

jacobian(f, u, p, t) = ForwardDiff.jacobian(u -> f(u, p, t), u)

function R(ρ, order, ::Val{MAX_ORDER}) where MAX_ORDER
    M = zero(MMatrix{MAX_ORDER, MAX_ORDER, typeof(ρ)})
    for r in 1:order, j in 1:order
        M[j, r] = prod(m->m - r*ρ, 0:j-1)/factorial(j)
    end
    SMatrix(M)
end

function change_step_size!(state::QNDFState{MAX_ORDER}) where MAX_ORDER
    state.n_constant_steps = 0
    @unpack f, order, dt, dtprev, U, RU, ∇, ∇tmp, γs, J, destats = state
    RU = R(dt/dtprev, order, Val(MAX_ORDER)) * U # Lower triangular
    @views mul!(∇tmp[:, 1:order], ∇[:, 1:order], RU[1:order, 1:order])
    ∇tmp, ∇ = ∇, ∇tmp # double buffering
    γdt = γs[order] * dt
    W = lu!(f.mass_matrix - γdt * J); destats.nw += 1
    @pack! state = ∇tmp, ∇, γdt, W
    return
end

function error_estimate(state::QNDFState, err, ε)
    @unpack t, uprev, u, opts = state
    @unpack internalnorm, reltol, abstol = opts
    atmp = DiffEqBase.calculate_residuals(err, uprev, u, abstol, reltol, internalnorm, t)
    ε * internalnorm(atmp, t)
end

"""
    $(SIGNATURES)
Return `q` such that `q * dt` is the step size estimate for the next step of
the `order+δ`-th order. When `δ==0`, `err` should be the error estimate, else,
it should be the error array.
"""
@inline function step_size_control(state::QNDFState, err, order, ::Val{δ}) where δ
    @unpack dt, εs = state
    @unpack step_bias, step_addon, qmin, qmax = state.opts
    EEst = δ == 0 ? err : error_estimate(state, err, εs[order + δ])
    η = step_bias[δ + 2]
    α = step_addon[δ + 2]
    exponent = inv(order + δ + 1)
    q = inv(η * EEst^exponent + α)
    min(qmax, max(qmin, q))
end

function choose_order_dt!(state::QNDFState{MAX_ORDER}) where MAX_ORDER
    @unpack order, dt, n_constant_steps, ∇, εs, tdir, EEst = state
    orderprev = order
    dtprev = dt
    n_constant_steps += 1
    q′ = step_size_control(state, EEst, order, Val(0))
    order′ = order
    # Wait for `order + 1` steps with the same `dt` and `order` before changing
    # order.
    #
    # LSODA:
    # https://github.com/lh3/misc/blob/cc0f36a9a19f35765efb9387389d9f3a6756f08f/math/lsoda.c#L1660-L1670
    if n_constant_steps >= order + 1
        if order > 1
            qm1 = @views step_size_control(state, ∇[:, order], order, Val(-1))
            if qm1 > q′
                q′ = qm1
                order′ = order - 1
            end
        end
        if order < MAX_ORDER
            qp1 = @views step_size_control(state, ∇[:, order+2], order, Val(+1))
            if qp1 > q′
                q′ = qp1
                order′ = order + 1
            end
        end
    end
    # LSODA
    # https://github.com/lh3/misc/blob/cc0f36a9a19f35765efb9387389d9f3a6756f08f/math/lsoda.c#L1754-L1756
    if q′ >= 1.1 # only change step size or order when `q` is sufficiently large
        dt *= q′
        order = order′
    end
    @pack! state = orderprev, dtprev, dt, order, n_constant_steps
    return nothing
end

function perform_step!(state::QNDFState)
    # read-only variables
    @unpack f, p, t, tend, γs, εs, 𝔹, order, opts, uprev, tdir, dtmin = state
    @unpack mass_matrix = f
    @unpack newton_max_iters, reltol, abstol, internalnorm = opts
    converge = false
    while !converge
        # write variables
        @unpack W, ∇, Ψ, isfinal, dt, predictor, u, ∇p1, γdt, destats = state
        copyto!(state.γ𝔹, γs[order] * 𝔹)
        @views mul!(Ψ, ∇[:, 1:order], state.γ𝔹[1:order])
        tn = isfinal ? tend : t + dt
        dt = tn - t
        @pack! state = tn, dt
        for i in eachindex(predictor)
            s = zero(eltype(∇))
            for j in 1:order
                s += ∇[i, j]
            end
            predictor[i] = uprev[i] + s
        end
        copyto!(u, predictor)
        fill!(∇p1, zero(eltype(∇p1)))

        ### Newton
        θ = η = nΔ = 1.0
        for iter in 1:newton_max_iters # Section 2.3
            residual = γdt * f(u, p, tn) - mass_matrix * (Ψ + ∇p1)
            Δ = W \ residual
            destats.nsolve += 1
            @. ∇p1 += Δ
            @. u = predictor + ∇p1
            nΔprev = nΔ
            destats.nnonliniter += 1
            @pack! state = u, uprev # just to make sure
            nΔ = error_estimate(state, Δ, true)
            iter > 1 && (θ = nΔ / nΔprev)
            θ > 2 && break

            iter > 1 && (η = θ / (1 - θ))
            if (iter == 1 && nΔ < 1e-5) || (iter > 1 && (η >= zero(η) && η * nΔ < 0.001))
                converge = true
                break
            end
        end
        converge && break
        debugmode() && @info "Newton Diverge"
        destats.nnonlinconvfail += 1
        abs(dt) <= dtmin && error("dt=dtmin=$dtmin, yet Newton still doesn't converge.")
        dtprev = dt
        dt = tdir * max(dtmin, 0.1abs(dt))
        isfinal = false
        @pack! state = dtprev, dt, isfinal
        change_step_size!(state)
    end # Newton

    state.EEst = error_estimate(state, state.∇p1, εs[order])
    return
end

function loopheader!(state)
    @unpack f, uprev, p, t, dt, tend, tdir, destats = state
    dtmin = eps(t)
    dtmax = abs(tend - t)
    dt = tdir * min(dtmax, max(dtmin, abs(dt)))
    isfinal = tdir * (tend - (t + dt)) < 10dtmin
    isfinal && (dt = tend - t)
    @pack! state = dtmin, dtmax, dt, isfinal
    # no Jacobian reuse
    state.J = jacobian(f, uprev, p, t); destats.njacs += 1
    if state.orderprev != state.order || state.dtprev != dt
        change_step_size!(state)
    end
    return
end

function step_till_accept!(state)
    rejections = 0
    while true
        perform_step!(state)
        @unpack ∇, isfinal, dt, dtmin, tdir, ∇p1, orderprev, order, EEst, destats = state
        debugmode() && @printf "[EEst: %.3f] [order: %1d] [t: %.5e] [dt: %.5e] [consts: %d]\n" EEst order state.t dt state.n_constant_steps
        if EEst <= one(EEst)
            destats.naccept += 1
            break
        end
        # reject
        debugmode() && @info "Error reject"
        destats.nreject += 1
        orderprev = order
        dtprev = dt
        if rejections >= 1 # If this step was rejected at least once already
            dt = tdir * max(dtmin, 0.1abs(dt)) # This is unexpected, so reduce the step size drastically
        else # use the controller
            q′ = step_size_control(state, EEst, order, Val(0))
            dt *= q′
        end
        rejections += 1
        isfinal = false
        @pack! state = orderprev, order, dtprev, dt
        change_step_size!(state)
    end

    @unpack order, ∇, ∇p1 = state
    @views @. ∇[:, order+2] = ∇p1 - ∇[:, order+1]
    @views @. ∇[:, order+1] = ∇p1
    for j in order:-1:1
        @views @. ∇[:, j] = ∇[:, j] + ∇[:, j+1]
    end
    t = state.tn
    @pack! state = t, ∇
    return
end

function solve!(state::QNDFState{MAX_ORDER}) where MAX_ORDER
    state.isfinal = false
    integ_iter = 0

    ts = [state.t]
    us = [state.uprev]

    while state.tdir * (state.tend - state.t) > 0
        integ_iter += 1
        loopheader!(state)

        step_till_accept!(state)

        choose_order_dt!(state)

        @unpack t, u = state
        push!(ts, t); push!(us, copy(u))
        uprev = copy(u)
        u = copy(uprev)
        @pack! state = u, uprev
    end
    return ts, us
end

qndf(args...; kwargs...) = solve!(init(args...; kwargs...))

function lorenz(u,p,t)
    [10.0(u[2]-u[1])
     u[1]*(28.0-u[3]) - u[2]
     u[1]*u[2] - (8/3)*u[3]]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0, 100.0)
prob = ODEProblem{false}(lorenz,u0,tspan)
ts, us = qndf(prob, 1e-6)
using Test
length(ts) < 6000
using Plots
display(plot3d([map(x->x[j], us) for j in 1:3]...))

#=
function rober(u,p,t)
y₁,y₂,y₃ = u
k₁,k₂,k₃ = p
[-k₁*y₁+k₃*y₂*y₃
k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
k₂*y₂^2]
end
prob = ODEProblem{false}(rober,[1.0,0.0,0.0],(0.0,1e5),[0.04,3e7,1e4])
ts, us = qndf(prob, 1e-6, abstol=1e-10, reltol=1e-10)
display(plot(ts[10:end], hcat(us...)'[10:end, 2], lab=false, xscale=:log10))
=#
