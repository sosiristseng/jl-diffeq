# # ComponentArrays
# https://github.com/SciML/ComponentArrays.jl are array blocks that can be accessed through a named index.
# You can compose differential equations without ModelingToolkit.
using ComponentArrays
using OrdinaryDiffEq
using SimpleUnPack: @unpack

tspan = (0.0, 20.0)

## Lorenz system
function lorenz!(D, u, p, t; f=0.0)
    @unpack σ, ρ, β = p
    @unpack x, y, z = u

    D.x = σ*(y - x)
    D.y = x*(ρ - z) - y - f
    D.z = x*y - β*z
    return nothing
end

lorenz_p = (σ=10.0, ρ=28.0, β=8/3)
lorenz_ic = ComponentArray(x=1.0, y=0.0, z=0.0)
lorenz_prob = ODEProblem(lorenz!, lorenz_ic, tspan, lorenz_p)


## Lotka-Volterra system
function lotka!(D, u, p, t; f=0.0)
    @unpack α, β, γ, δ = p
    @unpack x, y = u

    D.x =  α*x - β*x*y + f
    D.y = -γ*y + δ*x*y
    return nothing
end

lotka_p = (α=2/3, β=4/3, γ=1.0, δ=1.0)
lotka_ic = ComponentArray(x=1.0, y=1.0)
lotka_prob = ODEProblem(lotka!, lotka_ic, tspan, lotka_p)

## Composed Lorenz and Lotka-Volterra system
function composed!(D, u, p, t)
    c = p.c #coupling parameter
    @unpack lorenz, lotka = u

    lorenz!(D.lorenz, lorenz, p.lorenz, t, f=c*lotka.x)
    lotka!(D.lotka, lotka, p.lotka, t, f=c*lorenz.x)
    return nothing
end

comp_p = (lorenz=lorenz_p, lotka=lotka_p, c=0.01)
comp_ic = ComponentArray(lorenz=lorenz_ic, lotka=lotka_ic)
comp_prob = ODEProblem(composed!, comp_ic, tspan, comp_p)

# Solve problem
# We can solve the composed system...
comp_sol = solve(comp_prob)

#---
map(comp_sol(0:0.1:20).u) do u
    u.lotka.x
end

# ...or we can unit test one of the component systems
lotka_sol = solve(lotka_prob)
lotka_sol(1.0).x

# Symbolic mapping in `ODEFunction` using `syms=...`
# Only available in non-nested ComponentArrays
lotka_func = ODEFunction(lotka!; syms=keys(lotka_ic))
lab_sol = solve(ODEProblem(lotka_func, lotka_ic, tspan, lotka_p))
lab_sol(1.0, idxs=:x)
