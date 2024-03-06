#===

# ModelingToolkit: define ODEs with symbolic expressions

You can also define ODE systems symbolically using [ModelingToolkit.jl (MTK)](https://github.com/SciML/ModelingToolkit.jl) and let MTK generate high-performace ODE functions.

- transforming and simplifying expressions for performance
- Jacobian and Hessian generation
- accessing eliminated states (called observed variables)
- building and connecting subsystems programatically (component-based modeling)

See also [Simulating Big Models in Julia with ModelingToolkit @ JuliaCon 2021 Workshop](https://youtu.be/HEVOgSLBzWA).

===#

# ## Radioactive decay
# Here we use the same example of decaying radioactive elements

using ModelingToolkit
using DifferentialEquations
using Plots
using DisplayAs: PNG

#---
## independent variable (time) and dependent variables
@variables t c(t) RHS(t)

## parameters: decay rate
@parameters λ

## Differential operator w.r.t. time
D = Differential(t)

# Equations in MTK use the tilde character (`~`) for equality.
# Every MTK system requires a name. The `@named` macro simply ensures that the symbolic name matches the name in the REPL.

eqs = [
    RHS ~ -λ * c
    D(c) ~ RHS
]

@named sys = ODESystem(eqs, t)

# `structural_simplify()` simplifies the two equations to one.
# You can also use `@mtkbuild sys = ODESystem(eqs, t)` to merge these two steps into one
sys = structural_simplify(sys)

# Setup initial conditions, time span, parameter values, the `ODEProblem`, and solve the problem.
p = [λ => 1.0]
u0 = [c => 1.0]
tspan = (0.0, 2.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob)

# Visualize the solution with `Plots.jl`.
plot(sol, label="Exp decay") |> PNG

# The solution interface provides symbolic access. So you can access the results of `c` directly.
sol[c]

#---
sol(0.0:0.1:2.0, idxs=c)

# The eliminated term (RHS in this example) is still tracible.
plot(sol, idxs=[c, RHS], legend=:right) |> PNG

# The interface allows symbolic calculations.
plot(sol, idxs=[c*1000]) |> PNG

#===
## Lorenz system

We use the same Lorenz system example as above. Here we setup the initial conditions and parameters with default values.
===#

@variables t x(t)=1.0 y(t)=0.0 z(t)=0.0
@parameters (σ=10.0, ρ=28.0, β=8/3)

D = Differential(t)

eqs = [
    D(x) ~ σ * (y - x)
    D(y) ~ x * (ρ - z) - y
    D(z) ~ x * y - β * z
]

@mtkbuild sys = ODESystem(eqs, t)

# Here we are using default values, so we pass empty arrays for initial conditions and parameter values.
tspan = (0.0, 100.0)
prob = ODEProblem(sys, [], tspan, [])
sol = solve(prob)

# Plot the solution with symbols instead of index numbers.
plot(sol, idxs=(x, y, z), size=(400,400)) |> PNG

# ## Non-autonomous ODEs
# Sometimes a model might have a time-variant external force, which is too complex or impossible to express it symbolically. In such situation, one could apply `@register_symbolic` to it to exclude it from symbolic transformations and use it numberically.

@variables t x(t) f(t)
@parameters τ
D = Differential(t)

# Define a time-dependent random external force
value_vector = randn(10)

f_fun(t) = t >= 10 ? value_vector[end] : value_vector[Int(floor(t))+1]

# "Register" arbitrary Julia functions to be excluded from symbolic transformations. Just use it as-is (numberically).
@register_symbolic f_fun(t)
@named fol_external_f = ODESystem([f ~ f_fun(t), D(x) ~ (f - x)/τ], t)

#---
sys = structural_simplify(fol_external_f)
prob = ODEProblem(sys, [x => 0.0], (0.0, 10.0), [τ => 0.75])
sol = solve(prob)
plot(sol, idxs=[x, f]) |> PNG

# ## Second-order ODE systems
# `ode_order_lowering(sys)` automatically transforms a second-order ODE into two first-order ODEs.

using Plots
using ModelingToolkit
using DifferentialEquations

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [
    D(D(x)) ~ σ * (y-x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z
]

# `@mtkbuild` automatically does the transformations and simplifications
# The same as `sys = sys |> ode_order_lowering |> structural_simplify`
@mtkbuild sys = ODESystem(eqs, t)

# Note that one needs to provide the initial condition for x's derivative.
u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8/3
]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac=true)
sol = solve(prob)
plot(sol, idxs=(x, y, z), label="Trajectory", size=(500,500)) |> PNG

# ## Composing systems
# By defining connection equation(s) to couple ODE systems together, we can build component-based, hierarchical models.

using Plots
using ModelingToolkit
using DifferentialEquations

@parameters σ ρ β
@variables t x(t) y(t) z(t)

D = Differential(t)

eqs = [
    D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z
]

@named lorenz1 = ODESystem(eqs)
@named lorenz2 = ODESystem(eqs)

# Define relations (connectors) between the two systems.
@variables a(t)
@parameters γ

connections = [0 ~ lorenz1.x + lorenz2.y + a * γ]

@named connLorenz = ODESystem(connections, t , [a], [γ], systems = [lorenz1, lorenz2])

# All state variables in the combined system
states(connLorenz)

#---
u0 = [
    lorenz1.x => 1.0, lorenz1.y => 0.0, lorenz1.z => 0.0,
    lorenz2.x => 0.0, lorenz2.y => 1.0, lorenz2.z => 0.0,
    a => 2.0
]

p = [
    lorenz1.σ => 10.0, lorenz1.ρ => 28.0, lorenz1.β => 8/3,
    lorenz2.σ => 10.0, lorenz2.ρ => 28.0, lorenz2.β => 8/3,
    γ => 2.0
]

tspan = (0.0, 100.0)
sys = connLorenz |> structural_simplify
sol = solve(ODEProblem(sys, u0, tspan, p, jac=true))
plot(sol, idxs=(a, lorenz1.x, lorenz2.x), size=(600,600)) |> PNG

#===

### Convert existing functions into MTK systems

`modelingtoolkitize(prob)` generates MKT systems from regular DE problems. I t can also generate analytic Jacobin functions for faster solving.

Example: **[DAE index reduction](https://mtk.sciml.ai/stable/mtkitize_tutorials/modelingtoolkitize_index_reduction/)** for the pendulum problem, which cannot be solved by regular ODE solvers.

===#

using Plots
using ModelingToolkit
using DifferentialEquations
using LinearAlgebra

#---
function pendulum!(du, u, p, t)
    x, dx, y, dy, T = u
    g, L = p
    du[1] = dx
    du[2] = T*x
    du[3] = dy
    du[4] = T*y - g
    ## Do not write your function like this after you've learned MTK
    du[5] = x^2 + y^2 - L^2
    return nothing
end

#---
pendulum_fun! = ODEFunction(pendulum!, mass_matrix = Diagonal([1, 1, 1, 1, 0]))
u0 = [1.0, 0.0, 0.0, 0.0, 0.0]
p = [9.8, 1.0]
tspan = (0.0, 10.0)
pendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)

# Convert the ODE problem into a MTK system.
tracedSys = modelingtoolkitize(pendulum_prob)

# `structural_simplify()` and `dae_index_lowering()` transform the index-3 DAE into an index-0 ODE.
pendulumSys = tracedSys |> dae_index_lowering |> structural_simplify

# The default `u0` is included in the system already so one can use an empty array `[]` as the initial conditions.
prob = ODAEProblem(pendulumSys, [], tspan)
sol = solve(prob, abstol=1e-8, reltol=1e-8)
plot(sol, idxs=states(tracedSys)) |> PNG
