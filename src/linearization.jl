struct ZeroOrderHoldLinearization{Dx,Du,T,TA<:SMatrix{Dx,Dx},TB<:SMatrix{Dx,Du},Tc<:SVector{Dx}}
    dt::T
    A::TA
    B::TB
    c::Tc
end
struct FirstOrderHoldLinearization{Dx,Du,T,TA<:SMatrix{Dx,Dx},TB0<:SMatrix{Dx,Du},TBf<:SMatrix{Dx,Du},Tc<:SVector{Dx}}
    dt::T
    A::TA
    B0::TB0
    Bf::TBf
    c::Tc
end

linearize(f::LinearDynamics, x::StaticVector, u::StaticVector) = f
function linearize(f::LinearDynamics{Dx,Du}, x::StaticVector{Dx}, SC::StepControl{Du}) where {Dx,Du}
    eᴬᵗ, ∫eᴬᵗB = integrate_expAt_B(f.A, f.B, SC.t)
    _, ∫eᴬᵗc = integrate_expAt_B(f.A, f.c, SC.t)
    ZeroOrderHoldLinearization(SC.t, eᴬᵗ, ∫eᴬᵗB, ∫eᴬᵗc)
end
function linearize(f::LinearDynamics{Dx,Du}, x::StaticVector{Dx}, RC::RampControl{Du}) where {Dx,Du}
    eᴬᵗ, ∫eᴬᵗB, ∫eᴬᵗBtdt⁻¹ = integrate_expAt_Bt_dtinv(f.A, f.B, SC.t)
    _, ∫eᴬᵗc = integrate_expAt_B(f.A, f.c, SC.t)
    FirstOrderHoldLinearization(RC.t, eᴬᵗ, ∫eᴬᵗBtdt⁻¹, ∫eᴬᵗB - ∫eᴬᵗBtdt⁻¹, ∫eᴬᵗc)
end

function linearize(f::DifferentialDynamics, x::State, u::Control)
    A = ForwardDiff.jacobian(y -> f(y, u), x)
    B = ForwardDiff.jacobian(w -> f(x, w), u)
    A, B, f(x,u) - A*x - B*u
end
function linearize(f::DifferentialDynamics, x::StaticVector, SC::StepControl)
    A = ForwardDiff.jacobian(y -> propagate(f, y, SC), x)
    B = ForwardDiff.jacobian(w -> propagate(f, x, StepControl(SC.t, w)), SC.u)
    ZeroOrderHoldLinearization(SC.t, A, B, propagate(f, x, SC) - A*x - B*SC.u)
end
function linearize(f::DifferentialDynamics, x::StaticVector, RC::RampControl)
    A  = ForwardDiff.jacobian(y -> propagate(f, y, RC), x)
    B0 = ForwardDiff.jacobian(w -> propagate(f, x, RampControl(RC.t, w, RC.uf)), RC.u0)
    Bf = ForwardDiff.jacobian(w -> propagate(f, x, RampControl(RC.t, RC.u0, w)), RC.uf)
    FirstOrderHoldLinearization(RC.t, A, B0, Bf, propagate(f, x, RC) - A*x - B0*RC.u0 - Bf*RC.uf)
end

function (f::ZeroOrderHoldLinearization{Dx,Du})(x::StaticVector{Dx}, SC::StepControl{Du}) where {Dx,Du}
    @assert f.dt == SC.t
    f.A*x + f.B*SC.u + f.c
end
function (f::FirstOrderHoldLinearization{Dx,Du})(x::StaticVector{Dx}, RC::RampControl{Du}) where {Dx,Du}
    @assert f.dt == RC.t
    f.A*x + f.B0*RC.u0 + f.Bf*RC.uf + f.c
end
(f::FirstOrderHoldLinearization{Dx,Du})(x::StaticVector{Dx}, SC::StepControl{Du}) where {Dx,Du} = f(RampControl(SC))