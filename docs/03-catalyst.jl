# # Catalyst.jl
# `Catalyst.jl` is a domain specific language (DSL) for high performance simulation and modeling of chemical reaction networks. `Catalyst.jl` is based on `ModelingToolkit.jl` for symbolic transformations and generating model code.
# This part is lalrgely based on [Catalyst.jl Docs](https://docs.sciml.ai/Catalyst/dev/).
# ## Repressilator
# [Repressilator](https://en.wikipedia.org/wiki/Repressilator) model with three components in a negative feedback loop.

using Catalyst
using ModelingToolkit
using DifferentialEquations
using Plots
using DisplayAs: PNG

repressilator = @reaction_network begin
    hillr(P₃,α,K,n), ∅ --> m₁
    hillr(P₁,α,K,n), ∅ --> m₂
    hillr(P₂,α,K,n), ∅ --> m₃
    (δ,γ), m₁ ↔ ∅
    (δ,γ), m₂ ↔ ∅
    (δ,γ), m₃ ↔ ∅
    β, m₁ --> m₁ + P₁
    β, m₂ --> m₂ + P₂
    β, m₃ --> m₃ + P₃
    μ, P₁ --> ∅
    μ, P₂ --> ∅
    μ, P₃ --> ∅
end

# State variables in the reaction network
states(repressilator)

# Parameters in the reaction network
parameters(repressilator)

# Reactions in the reaction network
reactions(repressilator)

# ### Convert to an ODE system
odesys = convert(ODESystem, repressilator)

# To setup parameters (`ps`) and initial conditions (`u₀`), you can use Julia symbols to map the values.
p = [:α => .5, :K => 40, :n => 2, :δ => log(2)/120, :γ => 5e-3, :β => 20*log(2)/120, :μ => log(2)/60]
u₀ = [:m₁ => 0., :m₂ => 0., :m₃ => 0., :P₁ => 20.,:P₂ => 0.,:P₃ => 0.]

# Or you can also use MTK symbols from the ODE system with the `@unpack` macro (recommended)
@unpack m₁, m₂, m₃, P₁, P₂, P₃, α, K, n, δ, γ, β, μ = repressilator
p = [α => .5, K => 40, n => 2, δ => log(2)/120, γ => 5e-3, β => 20*log(2)/120, μ => log(2)/60]
u₀ = [m₁ => 0., m₂ => 0., m₃ => 0., P₁ => 20., P₂ => 0., P₃ => 0.]

# Then we can solve this reaction network
tspan = (0., 10000.)
oprob = ODEProblem(odesys, u₀, tspan, p)
sol = solve(oprob)
plot(sol) |> PNG

# Use extracted symbols for a phase plot.
plot(sol, idxs = (P₁, P₂)) |> PNG

# ### Convert to Stochastic Differential Equation (SDE) Models
# Convert the very same reaction network to Chemical Langevin Equation (CLE), adding Brownian motion terms to the state variables.
tspan = (0., 10000.)
sprob = SDEProblem(repressilator, u₀, tspan, p)

# solve and plot, `tstops` is used to specify enough points that the plot looks well-resolved.
sol = solve(sprob, LambaEulerHeun(), tstops=range(0.0, tspan[2]; length=1001))
plot(sol) |> PNG

# ### Convert to Gillespie's stochastic simulation
# Create a Gillespie stochastic simulation model from the same reaction network.
# The initial conditions should be integers
u₀ = [m₁ => 0, m₂ => 0, m₃ => 0, P₁ => 20, P₂ => 0, P₃ => 0]

# Create a discrete problem because our species are integer valued:
dprob = DiscreteProblem(repressilator, u₀, tspan, p)

# Create a `JumpProblem`, and specify Gillespie's Direct Method as the solver.
jprob = JumpProblem(repressilator, dprob, Direct(), save_positions=(false, false))

# Solve and visualize the problem.
sol = solve(jprob, SSAStepper(), saveat=10.)
plot(sol) |> PNG

#===
## Generating reaction systems programmatically

There are two ways to create a reaction for `ReactionSystem`s:
- `Reaction()` function
- `@reaction` macro
===#

@parameters α K n δ γ β μ
@variables t
@species m₁(t) m₂(t) m₃(t) P₁(t) P₂(t) P₃(t)

# ### Reaction function
# The `Reaction(rate, substrates, products)` function builds reactions. To allow for other stoichiometric coefficients we also provide a five argument form: `Reaction(rate, substrates, products, substrate_stoichiometries, product_stoichiometries)`

rxs = [
    Reaction(hillr(P₃,α,K,n), nothing, [m₁]),
    Reaction(hillr(P₁,α,K,n), nothing, [m₂]),
    Reaction(hillr(P₂,α,K,n), nothing, [m₃]),
    Reaction(δ, [m₁], nothing),
    Reaction(γ, nothing, [m₁]),
    Reaction(δ, [m₂], nothing),
    Reaction(γ, nothing, [m₂]),
    Reaction(δ, [m₃], nothing),
    Reaction(γ, nothing, [m₃]),
    Reaction(β, [m₁], [m₁,P₁]),
    Reaction(β, [m₂], [m₂,P₂]),
    Reaction(β, [m₃], [m₃,P₃]),
    Reaction(μ, [P₁], nothing),
    Reaction(μ, [P₂], nothing),
    Reaction(μ, [P₃], nothing)
]

# Use `ReactionSystem(reactions, indenpendeent_variable)` to collect these reactions. `@named` macro is used because every system in `ModelingToolkit` needs a name. `@named x = System(...)` is a short hand for `x = System(...; name=:x)`

@named repressilator = ReactionSystem(rxs, t)

# The `@reaction` macro provides the same syntax in the `@reaction_network` to build reactions. Note that `@reaction` macro only allows one-way reaction; **reversible arrows are not allowed**.

@variables t
@species P₁(t) P₂(t) P₃(t)

rxs = [(@reaction hillr($P₃,α,K,n), ∅ --> m₁),
       (@reaction hillr($P₁,α,K,n), ∅ --> m₂),
       (@reaction hillr($P₂,α,K,n), ∅ --> m₃),
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
       (@reaction μ, P₃ --> ∅)]

@named repressilator2 = ReactionSystem(rxs, t)

# ## Conservation laws
# We can use conservation laws to eliminate dependent variables. For example, in the chemical reaction `A + B <--> C`, given the initial concentrations of A, B, and C, the solver only needs to solve one state variable (either [A], [B], or [C]) instead of all three of them.

rn = @reaction_network begin
    (k₊,k₋), A + B <--> C
end

# initial condition and parameter values
setdefaults!(rn, [:A => 1.0, :B => 2.0, :C => 0.0, :k₊ => 1.0, :k₋ => 1.0])

# Let's convert it to a system of ODEs, using the conservation laws to eliminate two species, leaving only one of them as the state variable.
# The conserved quantities will be denoted as `Γ`s
osys = convert(ODESystem, rn; remove_conserved=true)

# Only one state variable (unknown) need to be solved
states(osys)

# The other two are constrained by conserved quantities
observed(osys)

# Sovle the problem
oprob = ODEProblem(osys, [], (0.0, 10.0), [])
sol = solve(oprob, Tsit5())

# You can still trace the eliminated variable
plot(sol, idxs=osys.C) |> PNG
