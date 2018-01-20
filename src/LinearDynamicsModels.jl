__precompile__()

module LinearDynamicsModels

using StaticArrays
using DifferentialDynamicsModels
using SymPy
using MacroTools

import DifferentialDynamicsModels: SteeringBVP, state_dim, control_dim
export LinearDynamics
export NIntegratorDynamics, DoubleIntegratorDynamics, TripleIntegratorDynamics

struct LinearDynamics{Dx,Du,T,DxDx,DxDu} <: DifferentialDynamics
    A::SMatrix{Dx,Dx,T,DxDx}
    B::SMatrix{Dx,Du,T,DxDu}
    c::SVector{Dx,T}
end

state_dim(::LinearDynamics{Dx,Du}) where {Dx,Du} = Dx
control_dim(::LinearDynamics{Dx,Du}) where {Dx,Du} = Du

(f::LinearDynamics{Dx,Du})(x::StaticVector{Dx}, u::StaticVector{Du}) where {Dx,Du} = f.A*x + f.B*u + f.c

function NIntegratorDynamics(n::Int, d::Int, ::Type{T} = Rational{Int}) where {T}    # notes: precision belongs to x and u, not f
    A = SMatrix{n*d,n*d,T}(diagm(ones(T, d*(n-1)), d))
    B = SMatrix{n*d,d,T}([zeros(T, d*(n-1), d); eye(T, d)])
    c = zeros(SVector{n*d,T})
    LinearDynamics(A,B,c)
end
DoubleIntegratorDynamics(d::Int, ::Type{T} = Rational{Int}) where {T} = NIntegratorDynamics(2, d, T)
TripleIntegratorDynamics(d::Int, ::Type{T} = Rational{Int}) where {T} = NIntegratorDynamics(3, d, T)

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

function SteeringBVP(f::LinearDynamics{Dx,Du,T},
                     j::TimePlusQuadraticControl{Du,T} = TimePlusQuadraticControl(eye(SMatrix{Du,Du,T}))) where {Dx,Du,T}
    SteeringBVP(f, j, EmptySteeringParams(), LinearQuadraticHelpers(f.A, f.B, f.c, j.R))
end

function (bvp::SteeringBVP{D,C})(x0::StaticVector{Dx,T},
                                 xf::StaticVector{Dx,T},
                                 c_max::T) where {Dx,Du,T<:AbstractFloat,D<:LinearDynamics{Dx,Du},C<:TimePlusQuadraticControl{Du}}
    x0 == xf && return T(0), BVPControl(T(0), xf, bvp.cache.x, bvp.cache.u)
    t = optimal_time(bvp, x0, xf, c_max)
    return bvp.cache.cost(x0, xf, t), BVPControl(t, xf, bvp.cache.x, bvp.cache.u)
end

function LinearQuadraticHelpers(A_::AbstractMatrix{T},
                                B_::AbstractMatrix{T},
                                c_::AbstractVector{T},
                                R_::AbstractMatrix{T} = eye(T, size(B_, 2))) where {T}
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
    sarray = !(A_ isa Array)
    t_args = :((t::T) where T<:AbstractFloat)
    xyt_args = :((x::AbstractVector{T}, y::AbstractVector{T}, t::T) where T<:AbstractFloat)
    xyts_args = :((x::AbstractVector{T}, y::AbstractVector{T}, t::T, s::T) where T<:AbstractFloat)
    LinearQuadraticHelpers(
        code2func(sympy2code.(symbolic_exprs["Ginv"], symbol_dict), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["expAt"], symbol_dict), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["cdrift"], symbol_dict), t_args, sarray),
        code2func(sympy2code.(symbolic_exprs["cost"], symbol_dict), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["dcost"], symbol_dict), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["ddcost"], symbol_dict), xyt_args, sarray),
        code2func(sympy2code.(symbolic_exprs["x_s"], symbol_dict), xyts_args, sarray),
        code2func(sympy2code.(symbolic_exprs["u_s"], symbol_dict), xyts_args, sarray),
        symbolic_exprs
    )
end

function sympy2code(x, symbol_dict = Dict())
    expr = parse(sympy_meth(:julia_code, x))
    MacroTools.postwalk(x -> x isa AbstractFloat ? :(T($x)) : get(symbol_dict, x, x), expr)
end

code2func(code, args, static_array = true) = eval(:($args -> $code))
function code2func(code::AbstractVector, args, static_array = true)
    N = length(code)
    if static_array
        body = :(SVector{$N}($(code...)))
    else
        body = :($T[$(code...)])
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

function topt_golden_section(c, x0::AbstractVector{T}, xf::AbstractVector{T}, t_max::T, ϵ::T = T(1e-3)) where {T<:AbstractFloat}
    gr = (1 + sqrt(T(5)))/2
    b = t_max
    a = t_max / 100
    ca, cb = c(x0, xf, a), c(x0, xf, b)
    while ca < cb; a /= 2; ca = c(x0, xf, a); end
    m1 = b - (b - a)/gr
    m2 = a + (b - a)/gr
    while abs(m1 - m2) > ϵ
        if c(x0, xf, m1) < c(x0, xf, m2)
            b = m2
        else
            a = m1
        end
        m1 = b - (b - a)/gr
        m2 = a + (b - a)/gr
    end
    (a + b)/2
end
function topt_bisection(dc, x0::AbstractVector{T}, xf::AbstractVector{T}, t_max::T, ϵ::T = T(1e-3)) where {T<:AbstractFloat}
    # Bisection
    # Broken for Float32 triple integrator with R = [1f-3],
    # q0 = SVector(-6.094263f0, 0.014560595f0, -0.6846263f0)
    # qf = SVector(-6.0940447f0, 0.0f0, 0.0f0)
    dc(x0, xf, t_max) < 0 && return t_max
    b = t_max
    a = t_max / 100
    dcval = dc(x0, xf, a)
    while dcval > 0
        a /= 2
        dcval = dc(x0, xf, a)
        (a == 0 || isnan(dcval) || isinf(dcval)) && return T(-1)    # workaround for above
    end
    m = T(0)
    dcval = T(1)
    while abs(dcval) > ϵ && abs(a - b) > ϵ
        m = (a + b)/2
        dcval = dc(x0, xf, m)
        dcval > 0 ? b = m : a = m
    end
    m
end
function topt_newton(dc, ddc, x0::AbstractVector{T}, xf::AbstractVector{T}, t_max::T, ϵ::T = T(1e-6)) where {T<:AbstractFloat}
    # Bisection / Newton's method combo
    dc(x0, xf, t_max) < 0 && return t_max
    b = t_max
    a = t_max / 100
    dcval = dc(x0, xf, a)
    while dcval > 0
        a /= 2
        dcval = dc(x0, xf, a)
        (a == 0 || isnan(dcval) || isinf(dcval)) && return T(-1)
    end
    # while dc(x0, xf, a) > 0; a /= 2; end
    t = t_max / 2
    dcval = dc(x0, xf, t)
    while abs(dcval) > ϵ && abs(a - b) > ϵ
        t = t - dcval / ddc(x0, xf, t)
        (t < a || t > b) && (t = (a+b)/2)
        dcval = dc(x0, xf, t)
        dcval > 0 ? b = t : a = t
    end
    t
end
function optimal_time(bvp::SteeringBVP{D,C},
                      x0::AbstractVector{T},
                      xf::AbstractVector{T},
                      t_max::T) where {D<:LinearDynamics,C<:TimePlusQuadraticControl,T<:AbstractFloat}
    t = (T === Float64 ? topt_newton(bvp.cache.dcost, bvp.cache.ddcost, x0, xf, t_max) :
                         topt_bisection(bvp.cache.dcost, x0, xf, t_max))
    t > 0 ? t : topt_golden_section(bvp.cache.cost, x0, xf, t_max)
end

end # module
