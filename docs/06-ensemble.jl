#===
# Parallel Ensemble Simulations

Docs: https://diffeq.sciml.ai/stable/features/ensemble/

## Solving an ODE With Different Initial Conditions

Solving $\dot{u} = 1.01u$ with $u(0)=0.5$ and $t \in [0, 1]$ with multiple initial conditions.
===#
using OrdinaryDiffEq
using Plots

# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))

# Define a new problem for each trajectory using `remake()`
# The initial conditions (u0) are changed in this example
function prob_func(prob, i, repeat)
    remake(prob, u0=rand() * prob.u0)
end

#===
The function could also capture the necessary data outside.

```julia
initial_conditions = range(0, stop=1, length=100)

function prob_func(prob, i, repeat)
  remake(prob, u0=initial_conditions[i])
end
```
===#

# Define an ensemble problem
ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)

# Ensemble simulations are solved by multithreading by default
sim = solve(ensemble_prob, trajectories=100)

# Each element in the results is an ODE solution
sim[1]

# You can plot all the results at once
plot(sim, linealpha=0.4)

# ## Solving an SDE with Different Parameters
using StochasticDiffEq
using Plots

function lotka_volterra!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end

function g!(du, u, p, t)
    du[1] = p[3] * u[1]
    du[2] = p[4] * u[2]
end

p = [1.5, 1.0, 0.1, 0.1]
prob = SDEProblem(lotka_volterra!, g!, [1.0, 1.0], (0.0, 10.0), p)

#---
function prob_func(prob, i, repeat)
    x = 0.3 * rand(2)
    remake(prob, p=[p[1:2]; x])
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sim = solve(ensemble_prob, SRIW1(), trajectories=10)

plot(sim, linealpha=0.5, color=:blue, idxs=(0, 1))
plot!(sim, linealpha=0.5, color=:red, idxs=(0, 2))

# Show the distribution of the ensemble solutions
summ = EnsembleSummary(sim, 0:0.1:10)
plot(summ, fillalpha=0.5)

# ## Ensemble simulations of Modelingtoolkit (MTK) models
# Radioactive decay example
using ModelingToolkit
using OrdinaryDiffEq
using Plots

@independent_variables t
@variables c(t) = 1.0
@parameters λ = 1.0
D = Differential(t)
@mtkbuild sys = ODESystem([D(c) ~ -λ * c], t)
prob = ODEProblem(sys, [], (0.0, 2.0), [])
solve(prob)

# Use the symbolic interface to change the parameter(s).
function changemtkparam(prob, i, repeat)
    ## Make a new copy of the parameter vector
    ## Ensure the changed will not affect the original ODE problem
    remake(prob, p=[λ => i * 0.5])
end

eprob = EnsembleProblem(prob, prob_func=changemtkparam)
sim = solve(eprob, trajectories=10)
plot(sim, linealpha=0.5)
