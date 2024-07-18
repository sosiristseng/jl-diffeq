md"""
# Generating reaction systems programmatically

There are two ways to create a reaction for a `ReactionSystem`:

- `Reaction()` function
- `@reaction` macro
"""

using Catalyst
using ModelingToolkit

@parameters α K n δ γ β μ
@variables t
@species m₁(t) m₂(t) m₃(t) P₁(t) P₂(t) P₃(t)

# ## Reaction function
# The `Reaction(rate, substrates, products)` function builds reactions.
# To allow for other stoichiometric coefficients we also provide a five argument form: `Reaction(rate, substrates, products, substrate_stoichiometries, product_stoichiometries)`

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

# Use `ReactionSystem(reactions, indenpendeent_variable)` to collect these reactions.
# `@named` macro is used because every system in `ModelingToolkit` needs a name.
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
