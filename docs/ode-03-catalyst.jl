#===
# Catalyst ODE examples

[Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) is a symbolic modeling package for analysis and high-performance simulation of chemical reaction networks.

## Repressilator

[Repressilator](https://en.wikipedia.org/wiki/Repressilator) model consists of a biochemical reaction network with three components in a negative feedback loop.
===#
using Catalyst
using ModelingToolkit
using OrdinaryDiffEq
using Plots

# Define the reaction network
repressilator = @reaction_network begin
    hillr(P₃, α, K, n), ∅ --> m₁
    hillr(P₁, α, K, n), ∅ --> m₂
    hillr(P₂, α, K, n), ∅ --> m₃
    (δ, γ), m₁ ↔ ∅
    (δ, γ), m₂ ↔ ∅
    (δ, γ), m₃ ↔ ∅
    β, m₁ --> m₁ + P₁
    β, m₂ --> m₂ + P₂
    β, m₃ --> m₃ + P₃
    μ, P₁ --> ∅
    μ, P₂ --> ∅
    μ, P₃ --> ∅
end

# Reactions in the reaction network
reactions(repressilator)

# State variables in the reaction network
unknowns(repressilator)

# Parameters in the reaction network
parameters(repressilator)

# To setup parameters (`p`) and initial conditions (`u0`), you can use Julia symbols to map the values.
p = [:α => 0.5, :K => 40, :n => 2, :δ => log(2) / 120, :γ => 5e-3, :β => 20 * log(2) / 120, :μ => log(2) / 60]
u0 = [:m₁ => 0.0, :m₂ => 0.0, :m₃ => 0.0, :P₁ => 20.0, :P₂ => 0.0, :P₃ => 0.0]

# Or you can also use symbols from the reaction system with the `@unpack` macro (less error prone)
@unpack m₁, m₂, m₃, P₁, P₂, P₃, α, K, n, δ, γ, β, μ = repressilator
p = [α => 0.5, K => 40, n => 2, δ => log(2) / 120, γ => 5e-3, β => 20 * log(2) / 120, μ => log(2) / 60]
u0 = [m₁ => 0.0, m₂ => 0.0, m₃ => 0.0, P₁ => 20.0, P₂ => 0.0, P₃ => 0.0]

# Then we can solve this reaction network as an ODE problem
tspan = (0.0, 10000.0)
oprob = ODEProblem(repressilator, u0, tspan, p)
sol = solve(oprob)
plot(sol)

# Use extracted symbols for a phase plot
plot(sol, idxs=(P₁, P₂))

#===
## Generating reaction systems programmatically

There are two ways to create a reaction for a `ReactionSystem`:

- `Reaction()` function.
- `@reaction` macro.

The `Reaction(rate, substrates, products)` function builds reactions.

To allow for other stoichiometric coefficients we also provide a five argument form: `Reaction(rate, substrates, products, substrate_stoichiometries, product_stoichiometries)`
===#
using Catalyst
using ModelingToolkit

@parameters α K n δ γ β μ
@variables t
@species m₁(t) m₂(t) m₃(t) P₁(t) P₂(t) P₃(t)

rxs = [
    Reaction(hillr(P₃, α, K, n), nothing, [m₁]),
    Reaction(hillr(P₁, α, K, n), nothing, [m₂]),
    Reaction(hillr(P₂, α, K, n), nothing, [m₃]),
    Reaction(δ, [m₁], nothing),
    Reaction(γ, nothing, [m₁]),
    Reaction(δ, [m₂], nothing),
    Reaction(γ, nothing, [m₂]),
    Reaction(δ, [m₃], nothing),
    Reaction(γ, nothing, [m₃]),
    Reaction(β, [m₁], [m₁, P₁]),
    Reaction(β, [m₂], [m₂, P₂]),
    Reaction(β, [m₃], [m₃, P₃]),
    Reaction(μ, [P₁], nothing),
    Reaction(μ, [P₂], nothing),
    Reaction(μ, [P₃], nothing)
]

# Use `ReactionSystem(reactions, indenpendeent_variable)` to collect these reactions. `@named` macro is used because every system in `ModelingToolkit.jl` needs a name.
# `@named x = System(...)` is a short hand for `x = System(...; name=:x)`
@named repressilator = ReactionSystem(rxs, t)

# The `@reaction` macro provides the same syntax in the `@reaction_network` to build reactions.
# Note that `@reaction` macro only allows one-way reaction; **reversible arrows are not allowed**.

@variables t
@species P₁(t) P₂(t) P₃(t)

rxs = [
    (@reaction hillr($P₃, α, K, n), ∅ --> m₁),
    (@reaction hillr($P₁, α, K, n), ∅ --> m₂),
    (@reaction hillr($P₂, α, K, n), ∅ --> m₃),
    (@reaction δ, m₁ --> ∅),
    (@reaction γ, ∅ --> m₁),
    (@reaction δ, m₂ --> ∅),
    (@reaction γ, ∅ --> m₂),
    (@reaction δ, m₃ --> ∅),
    (@reaction γ, ∅ --> m₃),
    (@reaction β, m₁ --> m₁ + P₁),
    (@reaction β, m₂ --> m₂ + P₂),
    (@reaction β, m₃ --> m₃ + P₃),
    (@reaction μ, P₁ --> ∅),
    (@reaction μ, P₂ --> ∅),
    (@reaction μ, P₃ --> ∅)
]

@named repressilator2 = ReactionSystem(rxs, t)

#===
## Conservation laws
We can use conservation laws to eliminate some unknown variables.
For example, in the chemical reaction `A + B <--> C`, given the initial concentrations of A, B, and C, the solver needs to find only one of [A], [B], and [C] instead of all three.
===#
using Catalyst
using ModelingToolkit
using OrdinaryDiffEq
using Plots

#---
rn = @reaction_network begin
    (k₊, k₋), A + B <--> C
end

# Set initial condition and parameter values
setdefaults!(rn, [:A => 1.0, :B => 2.0, :C => 0.0, :k₊ => 1.0, :k₋ => 1.0])

# Let's convert it to a system of ODEs, using the conservation laws to eliminate two species, leaving only one of them as the state variable.
# The conserved quantities will be denoted as `Γ`s
osys = convert(ODESystem, rn; remove_conserved=true) |> structural_simplify

# Only one (unknown) state variable need to be solved
unknowns(osys)

# The other two are constrained by conserved quantities
observed(osys)

# Solve the problem
oprob = ODEProblem(osys, [], (0.0, 10.0), [])
sol = solve(oprob, Tsit5())

# You can still trace the eliminated variable
plot(sol, idxs=osys.C)
