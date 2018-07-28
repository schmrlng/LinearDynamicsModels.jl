__precompile__()

module LinearDynamicsModels

using LinearAlgebra
using StaticArrays
using DifferentialDynamicsModels
using ForwardDiff
using SymPy
using MacroTools

import DifferentialDynamicsModels: SteeringBVP
import DifferentialDynamicsModels: state_dim, control_dim, duration, propagate, instantaneous_control
export LinearDynamics, ZeroOrderHoldLinearization, FirstOrderHoldLinearization, linearize
export NIntegratorDynamics, DoubleIntegratorDynamics, TripleIntegratorDynamics

include("utils.jl")

# Continous-Time Linear Time-Invariant Systems
struct LinearDynamics{Dx,Du,TA<:StaticMatrix{Dx,Dx},TB<:StaticMatrix{Dx,Du},Tc<:StaticVector{Dx}} <: DifferentialDynamics
    A::TA
    B::TB
    c::Tc
end

state_dim(::LinearDynamics{Dx,Du}) where {Dx,Du} = Dx
control_dim(::LinearDynamics{Dx,Du}) where {Dx,Du} = Du

(f::LinearDynamics{Dx,Du})(x::StaticVector{Dx}, u::StaticVector{Du}) where {Dx,Du} = f.A*x + f.B*u + f.c
function propagate(f::LinearDynamics{Dx,Du}, x::StaticVector{Dx}, SC::StepControl{Du}) where {Dx,Du}
    y = f.B*SC.u + f.c
    eᴬᵗ, ∫eᴬᵗy = integrate_expAt_B(f.A, y, SC.t)
    eᴬᵗ*x + ∫eᴬᵗy
end # @test propagate(f, x, SC) ≈ linearize(f, x, SC)(x, SC)
function propagate(f::LinearDynamics{Dx,Du}, x::StaticVector{Dx}, RC::RampControl{Du}) where {Dx,Du}
    y = f.B*RC.uf + f.c
    eᴬᵗ, ∫eᴬᵗy = integrate_expAt_B(f.A, y, RC.t)
    z = f.B*(RC.u0 - RC.uf)
    _, _, ∫eᴬᵗztdt⁻¹ = integrate_expAt_Bt(f.A, z. RC.t)
    eᴬᵗ*x + ∫eᴬᵗy + ∫eᴬᵗztdt⁻¹
end # @test propagate(f, x, RC) ≈ linearize(f, x, RC)(x, RC)

# Discrete-Time Linear Time-Invariant Systems
include("linearization.jl")

# NIntegrators (DoubleIntegrator, TripleIntegrator, etc.)
function NIntegratorDynamics(::Val{N}, ::Val{D}, ::Type{T} = Rational{Int}) where {N,D,T}
    A = diagm(Val(D) => ones(SVector{(N-1)*D,T}))
    B = [zeros(SMatrix{(N-1)*D,D,T}); SMatrix{D,D,T}(I)]
    c = zeros(SVector{N*D,T})
    LinearDynamics(A,B,c)
end
NIntegratorDynamics(N::Int, D::Int, ::Type{T} = Rational{Int}) where {T} = NIntegratorDynamics(Val(N), Val(D), T)
DoubleIntegratorDynamics(D::Int, ::Type{T} = Rational{Int}) where {T} = NIntegratorDynamics(2, D, T)
TripleIntegratorDynamics(D::Int, ::Type{T} = Rational{Int}) where {T} = NIntegratorDynamics(3, D, T)

# TimePlusQuadraticControl BVPs
function SteeringBVP(f::LinearDynamics{Dx,Du}, j::TimePlusQuadraticControl{Du};
                     compile::Union{Val{false},Val{true}}=Val(false)) where {Dx,Du}
    compile === Val(true) ? SteeringBVP(f, j, EmptySteeringConstraints(), LinearQuadraticHelpers(f.A, f.B, f.c, j.R)) :
                            SteeringBVP(f, j, EmptySteeringConstraints(), EmptySteeringCache())
end

## Ad-Hoc Steering
struct LinearQuadraticSteeringControl{Dx,Du,T,
                                      Tx0<:StaticVector{Dx},
                                      Txf<:StaticVector{Dx},
                                      TA<:StaticMatrix{Dx,Dx},
                                      TB<:StaticMatrix{Dx,Du},
                                      Tc<:StaticVector{Dx},
                                      TR<:StaticMatrix{Du,Du},
                                      Tz<:StaticVector{Dx}} <: ControlInterval
    t::T
    x0::Tx0
    xf::Txf
    A::TA
    B::TB
    c::Tc
    R::TR
    z::Tz
end
duration(lqsc::LinearQuadraticSteeringControl) = lqsc.t
propagate(f::LinearDynamics, x::State, lqsc::LinearQuadraticSteeringControl) = (x - lqsc.x0) + lqsc.xf
function propagate(f::LinearDynamics, x::State, lqsc::LinearQuadraticSteeringControl, s::Number)
    x0, A, B, c, R, z = lqsc.x0, lqsc.A, lqsc.B, lqsc.c, lqsc.R, lqsc.z
    eᴬˢ, ∫eᴬˢc = integrate_expAt_B(A, c, s)
    Gs = integrate_expAt_B_expATt(A, B*(R\B'), s)
    (x - x0) + eᴬˢ*x0 + ∫eᴬˢc + Gs*(eᴬˢ'\z)
end
function instantaneous_control(lqsc::LinearQuadraticSteeringControl, s::Number)
    A, B, R, z = lqsc.A, lqsc.B, lqsc.R, lqsc.z
    eᴬˢ = exp(A*s)
    (R\B')*(eᴬˢ'\z)
end

function (bvp::SteeringBVP{D,C,EmptySteeringConstraints,EmptySteeringCache})(x0::StaticVector{Dx},
                                                                             xf::StaticVector{Dx},
                                                                             c_max::T) where {Dx,Du,
                                                                                              T<:Number,
                                                                                              D<:LinearDynamics{Dx,Du},
                                                                                              C<:TimePlusQuadraticControl{Du}}
    f = bvp.dynamics
    j = bvp.cost
    A, B, c, R = f.A, f.B, f.c, j.R
    x0 == xf && return (cost=T(0), controls=LinearQuadraticSteeringControl(T(0), x0, xf, A, B, c, R, zeros(typeof(c))))
    t = optimal_time(bvp, x0, xf, c_max)
    Q = B*(R\B')
    G = integrate_expAt_B_expATt(A, Q, t)
    eᴬᵗ, ∫eᴬᵗc = integrate_expAt_B(A, c, t)
    x̄ = eᴬᵗ*x0 + ∫eᴬᵗc
    z = eᴬᵗ'*(G\(xf - x̄))
    (cost=cost(f, j, x0, xf, t), controls=LinearQuadraticSteeringControl(t, x0, xf, A, B, c, R, z))
end

function cost(f::LinearDynamics{Dx,Du}, j::TimePlusQuadraticControl{Du},
              x0::StaticVector{Dx}, xf::StaticVector{Dx}, t) where {Dx,Du}
    A, B, c, R = f.A, f.B, f.c, j.R
    Q = B*(R\B')
    G = integrate_expAt_B_expATt(A, Q, t)
    eᴬᵗ, ∫eᴬᵗc = integrate_expAt_B(A, c, t)
    x̄ = eᴬᵗ*x0 + ∫eᴬᵗc
    t + (xf - x̄)'*(G\(xf - x̄))
end

function dcost(f::LinearDynamics{Dx,Du}, j::TimePlusQuadraticControl{Du},
               x0::StaticVector{Dx}, xf::StaticVector{Dx}, t) where {Dx,Du}
    A, B, c, R = f.A, f.B, f.c, j.R
    Q = B*(R\B')
    G = integrate_expAt_B_expATt(A, Q, t)
    eᴬᵗ, ∫eᴬᵗc = integrate_expAt_B(A, c, t)
    x̄ = eᴬᵗ*x0 + ∫eᴬᵗc
    z = eᴬᵗ'*(G\(xf - x̄))
    1 - 2*(A*x0 + c)'*z - z'*Q*z
end

function optimal_time(bvp::SteeringBVP{D,C,EmptySteeringConstraints,EmptySteeringCache},
                      x0::StaticVector{Dx},
                      xf::StaticVector{Dx},
                      t_max::T) where {Dx,Du,T<:Number,D<:LinearDynamics{Dx,Du},C<:TimePlusQuadraticControl{Du}}
    t = bisection(t -> dcost(bvp.dynamics, bvp.cost, x0, xf, t), t_max/100, t_max)
    t !== nothing ? t : golden_section(cost, t_max/100, t_max)
end

## Compiled Steering Functions (using SymPy; returns `BVPControl`s)
struct LinearQuadraticHelpers{FGinv<:Function,
                              FexpAt<:Function,
                              Fcdrift<:Function,
                              Fcost<:Function,
                              Fdcost<:Function,
                              Fddcost<:Function,
                              Fx<:Function,
                              Fu<:Function} <: SteeringCache
    Ginv::FGinv
    expAt::FexpAt
    cdrift::Fcdrift
    cost::Fcost
    dcost::Fdcost
    ddcost::Fddcost
    x::Fx
    u::Fu
    symbolic_exprs::Dict{String,Union{Sym,Vector{Sym},Matrix{Sym}}}
end

function (bvp::SteeringBVP{D,C,EmptySteeringConstraints,<:LinearQuadraticHelpers})(x0::StaticVector{Dx},
                                                                                   xf::StaticVector{Dx},
                                                                                   c_max::T) where {Dx,Du,
                                                                                                    T<:Number,
                                                                                                    D<:LinearDynamics{Dx,Du},
                                                                                                    C<:TimePlusQuadraticControl{Du}}
    x0 == xf && return (cost=T(0), controls=BVPControl(T(0), x0, xf, bvp.cache.x, bvp.cache.u))
    t = optimal_time(bvp, x0, xf, c_max)
    (cost=bvp.cache.cost(x0, xf, t), controls=BVPControl(t, x0, xf, bvp.cache.x, bvp.cache.u))
end

function LinearQuadraticHelpers(A_::AbstractMatrix, B_::AbstractMatrix, c_::AbstractVector, R_::AbstractMatrix)
    A, B, c, R = Array(A_), Array(B_), Vector(c_), Array(R_)
    Dx, Du = size(B)
    t, s = symbols("t s", real=true)
    x = collect(symbols(join(("x$i" for i in 1:Dx), " "), real=true))
    y = collect(symbols(join(("y$i" for i in 1:Dx), " "), real=true))

    expAt = exp(A*t)
    expAs = exp(A*s)
    expAt_s = exp(A*(t - s))
    G = integrate(expAt*B*inv(R)*B'*expAt', t)
    Ginv = inv(G)
    cdrift = integrate(expAt, t)*c
    xbar = expAt*x + cdrift
    cost = t + (y - xbar)'*Ginv*(y - xbar)
    dcost = diff(cost, t)
    ddcost = diff(cost, t, 2)
    x_s = expAs*x + integrate(expAs, s)*c + integrate(expAs*B*inv(R)*B'*expAs', s)*expAt_s'*Ginv*(y-xbar)
    u_s = inv(R)*B'*expAt_s'*Ginv*(y-xbar)

    symbolic_exprs = Dict{String,Union{Sym,Vector{Sym},Matrix{Sym}}}(
        "Ginv" => Ginv,
        "expAt" => expAt,
        "cdrift" => cdrift,
        "cost" => cost,
        "dcost" => dcost,
        "ddcost" => ddcost,
        "x_s" => x_s,
        "u_s" => u_s,
        "t" => t,
        "s" => s,
        "x" => x,
        "y" => y
    )
    for (k, v) in symbolic_exprs
        symbolic_exprs[k] = collect.(expand.(v), t)
    end

    symbol_dict = merge(Dict(Symbol("x$i") => :(x[$i]) for i in 1:Dx),
                        Dict(Symbol("y$i") => :(y[$i]) for i in 1:Dx))
    sarray    = A_ isa StaticArray
    t_args    = :((t::T) where {T})
    xyt_args  = :((x::AbstractVector, y::AbstractVector, t::T) where {T})
    xyts_args = :((x::AbstractVector, y::AbstractVector, t::T, s) where {T})
    LinearQuadraticHelpers(
        code2func(sympy2code.(symbolic_exprs["Ginv"], Ref(symbol_dict)), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["expAt"], Ref(symbol_dict)), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["cdrift"], Ref(symbol_dict)), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["cost"], Ref(symbol_dict)), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["dcost"], Ref(symbol_dict)), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["ddcost"], Ref(symbol_dict)), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["x_s"], Ref(symbol_dict)), xyts_args, sarray),
        code2func(sympy2code.(symbolic_exprs["u_s"], Ref(symbol_dict)), xyts_args, sarray),
        symbolic_exprs
    )
end

function sympy2code(x, symbol_dict = Dict())
    code = foldl(replace, (".+" => " .+",
                           ".-" => " .-",
                           ".*" => " .*",
                           "./" => " ./",
                           ".^" => " .^"); init=sympy_meth(:julia_code, x))    # TODO: sympy upstream PR
    expr = Meta.parse(code)
    MacroTools.postwalk(x -> x isa AbstractFloat ? :(T($x)) : get(symbol_dict, x, x), expr)
end

code2func(code, args, static_array = true) = eval(:($args -> $code))
function code2func(code::AbstractVector, args, static_array = true)
    N = length(code)
    if static_array
        body = :(SVector{$N}($(code...)))
    else
        body = :([$(code...)])
    end
    eval(:($args -> $body))
end
function code2func(code::AbstractMatrix, args, static_array = true)
    M,N = size(code)
    if static_array
        body = :(SMatrix{$M,$N}($(code...)))
    else
        body = Expr(:vcat, (Expr(:row, code[i,:]...) for i in 1:M)...)
    end
    eval(:($args -> $body))
end

function optimal_time(bvp::SteeringBVP{D,C,EmptySteeringConstraints,<:LinearQuadraticHelpers},
                      x0::StaticVector{Dx},
                      xf::StaticVector{Dx},
                      t_max::T) where {Dx,Du,T<:Number,D<:LinearDynamics{Dx,Du},C<:TimePlusQuadraticControl{Du}}
    cost   = (s -> (Base.@_inline_meta; bvp.cache.cost(x0, xf, s)))    # closures and optimizers below both @inline-d
    dcost  = (s -> (Base.@_inline_meta; bvp.cache.dcost(x0, xf, s)))
    ddcost = (s -> (Base.@_inline_meta; bvp.cache.ddcost(x0, xf, s)))
    t = (T === Float64 ? newton(dcost, ddcost, t_max/100, t_max) :
                         bisection(dcost, t_max/100, t_max))
    t !== nothing ? t : golden_section(cost, t_max/100, t_max)
end

end # module
