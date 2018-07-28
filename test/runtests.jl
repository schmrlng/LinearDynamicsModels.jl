using LinearDynamicsModels
using DifferentialDynamicsModels
using StaticArrays
using Test

@maintain_type struct RoboState{T} <: FieldVector{2,T}
    x::T
    y::T
end

SI1 = SingleIntegratorDynamics{2}()
SI2 = NIntegratorDynamics(1, 2)
@test state_dim(SI2) == 2
@test control_dim(SI2) == 2

x0 = rand(RoboState{Float64})
sc = StepControl(.5, rand(SVector{2,Float64}))
rc = RampControl(.5, rand(SVector{2,Float64}), rand(SVector{2,Float64}))
@test propagate(SI1, x0, sc) ≈ propagate(SI2, x0, sc)
@test propagate(SI2, x0, sc) isa RoboState
@test propagate(SI1, x0, rc) ≈ propagate(SI2, x0, rc)
@test propagate(SI2, x0, rc) isa RoboState


# bvp1 = SteeringBVP(SI, Time(), constraints=BoundedControlNorm())
# bvp2 = SteeringBVP(SI, Time(), constraints=BoundedControlNorm())