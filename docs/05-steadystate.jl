#===
# Steady state solutions

- DiffEq docs: https://docs.sciml.ai/DiffEqDocs/stable/solvers/steady_state_solve/
- ModelingToolkit docs: https://docs.sciml.ai/ModelingToolkit/stable/tutorials/nonlinear/
- NonlinearSolve docs: https://docs.sciml.ai/NonlinearSolve/stable/

Solving steady state solutions for an ODE system is to find a combination of state variables such that their derivatives are all zeroes. Their are two ways:

- Running the ODE solver until a steady state is reached (`DynamicSS()` and `DifferentialEquations.jl`)
- Using a root-finding algorithm to find a steady state (`SSRootfind()` and `NonlinearSolve.jl`)
===#

# ## Defining a steady state problem
using SteadyStateDiffEq
using OrdinaryDiffEq

model(u, p, t) = 1 - p * u

p = 1.0
u0 = 0.0
prob = SteadyStateProblem(model, u0, p)
alg = DynamicSS(Tsit5())

# Solve the problem. The result should be close to 1.0.
sol = solve(prob, alg)

# ## Modelingtoolkit solving nonlinear systems
# Use `NonlinearSolve.jl` and `NonlinearSystem()`.
using ModelingToolkit
using NonlinearSolve

@variables x y z
@parameters σ ρ β

eqs = [
    0 ~ σ * (y - x),
    0 ~ x * (ρ - z) - y,
    0 ~ x * y - β * z
]

@mtkbuild ns = NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])

guess = [x => 1.0, y => 0.0, z => 0.0]
ps = [σ => 10.0, ρ => 26.0, β => 8 / 3]
prob = NonlinearProblem(ns, guess, ps)
sol = solve(prob, NewtonRaphson()) ## The results should be all zeroes

# Another nonlinear system example with `structural_simplify()`.
@independent_variables t
@variables u1(t) u2(t) u3(t) u4(t) u5(t)

eqs = [
    0 ~ u1 - sin(u5)
    0 ~ u2 - cos(u1)
    0 ~ u3 - hypot(u1, u2)
    0 ~ u4 - hypot(u2, u3)
    0 ~ u5 - hypot(u4, u1)
]

@mtkbuild sys = NonlinearSystem(eqs, [u1, u2, u3, u4, u5], [])

# You can simplify the problem using `structural_simplify()`.
# There will be only one state variable left. The solve can solve the problem faster.
prob = NonlinearProblem(sys, [u5 => 0.0])
sol = solve(prob, NewtonRaphson())

# The answer should be 1.6 and 1.0.
@show sol[u5] sol[u1];

# ## Finding Steady States through Homotopy Continuation
# For systems of rational polynomials. e.g., mass-action reactions and reaction rates with integer Hill coefficients.
# Source: https://docs.sciml.ai/Catalyst/stable/steady_state_functionality/homotopy_continuation/#homotopy_continuation
using Catalyst
import HomotopyContinuation

wilhelm_2009_model = @reaction_network begin
    k1, Y --> 2X
    k2, 2X --> X + Y
    k3, X + Y --> Y
    k4, X --> 0
end

# There are three steady states
ps = [:k1 => 8.0, :k2 => 2.0, :k3 => 1.0, :k4 => 1.5]
steady_states = hc_steady_states(wilhelm_2009_model, ps)

# ## Steady state stability computation
# `Catalyst.steady_state_stability()` shows there are two stable and one unstable steady states.
using Catalyst
[Catalyst.steady_state_stability(sstate, wilhelm_2009_model, ps) for sstate in steady_states]
