#===
# SDEs with ModelingToolkit.jl

Define a stochastic Lorentz system using `SDESystem(equations, noises, iv, dv, ps)`.

Add a diagonal noise with 10% of the magnitude, using a Brownian variable (`@brownian x`).
===#
using ModelingToolkit
using DifferentialEquations
using Plots

@parameters σ ρ β
@variables t x(t) y(t) z(t)
@brownian a
D = Differential(t)

eqs = [
    D(x) ~ σ * (y - x) + 0.1a * x,
    D(y) ~ x * (ρ - z) - y + 0.1a * y,
    D(z) ~ x * y - β * z + 0.1a * z
]

@mtkbuild de = System(eqs, t)

u0map = [
    x => 1.0,
    y => 0.0,
    z => 0.0
]

parammap = [
    σ => 10.0,
    β => 26.0,
    ρ => 2.33
]

tspan = (0.0, 100.0)

prob = SDEProblem(de, u0map, tspan, parammap)
sol = solve(prob, LambaEulerHeun())
plot(sol, idxs=(x, y, z))
