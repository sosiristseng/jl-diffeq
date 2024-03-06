#===

# Parallel Ensemble Simulations

Docs: https://diffeq.sciml.ai/stable/features/ensemble/

## Solving an ODE With Different Initial Conditions

Solving $\dot{u} = 1.01u$ with $u(0)=0.5$ and $t \in [0, 1]$

===#

using DifferentialEquations
using Plots
using DisplayAs: PNG

# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u,p,t)->1.01u, 0.5, (0.0, 1.0))

# Define a new problem for each trajectory using `remake()`
# The initial conditions (u0) are changed in this example
function prob_func(prob, i, repeat)
    remake(prob, u0=rand() * prob.u0)
end

#===

You could also obtain the necessary data from the outside of `prob_func()`

```julia
initial_conditions = range(0, stop=1, length=100)

function prob_func(prob, i, repeat)
  remake(prob, u0=initial_conditions[i])
end
```

===#

# Define an ennsemble problem
ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)

# Ensemble simulations use multithreading by default
sim = solve(ensemble_prob, trajectories=100)

# Each element in the result is an ODE solution
sim[1]

# You can plot all the results at once
plot(sim, linealpha=0.4) |> PNG

# ## Solving an SDE with Different Parameters
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
    remake(prob, p=[p[1:2];x])
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sim = solve(ensemble_prob, SRIW1(), trajectories=10)

fig = plot(sim, linealpha=0.6, color=:blue, idxs=(0,1))
plot!(fig, sim, linealpha=0.6, color=:red, idxs=(0,2))

# Show the distribution of the ensemble solutions
summ = EnsembleSummary(sim, 0:0.1:10)
plot(summ, fillalpha=0.5) |> PNG
