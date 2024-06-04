#===

# Optimization Problems

From: https://mtk.sciml.ai/dev/tutorials/optimization/

## 2D Rosenbrock Function

Wikipedia: https://en.wikipedia.org/wiki/Rosenbrock_function

Find $(x, y)$ that minimizes the loss function $(a - x)^2 + b(y - x^2)^2$

===#
using ModelingToolkit
using Optimization
using OptimizationOptimJL

#---
@variables begin
    x, [bounds = (-2.0, 2.0)]
    y, [bounds = (-1.0, 3.0)]
end

@parameters a = 1 b = 1

# Target (loss) function
loss = (a - x)^2 + b * (y - x^2)^2

# The OptimizationSystem
@named sys = OptimizationSystem(loss, [x, y], [a, b])

# MTK can generate Gradient and Hessian to solve the problem more efficiently.
u0 = [
    x => 1.0
    y => 2.0
]
p = [
    a => 1.0
    b => 100.0
]

prob = OptimizationProblem(sys, u0, p, grad=true, hess=true)

## The true solution is (1.0, 1.0)
sol = solve(prob, GradientDescent())

# ### Adding constraints
# `OptimizationSystem(..., constraints = cons)`

@variables begin
    x, [bounds = (-2.0, 2.0)]
    y, [bounds = (-1.0, 3.0)]
end

@parameters a = 1 b = 100

loss = (a - x)^2 + b * (y - x^2)^2
cons = [
    x^2 + y^2 ≲ 1,
]

@named sys = OptimizationSystem(loss, [x, y], [a, b], constraints=cons)

u0 = [x => 0.14, y => 0.14]
prob = OptimizationProblem(sys, u0, grad=true, hess=true, cons_j=true, cons_h=true)

# Use interior point Newton method for contrained optimization
solve(prob, IPNewton())

#===
## Parameter estimation

From: https://docs.sciml.ai/DiffEqParamEstim/stable/getting_started/

`DiffEqParamEstim.jl` is not installed with `DifferentialEquations.jl`. You need to install it manually:

```julia
using Pkg
Pkg.add("DiffEqParamEstim")
using DiffEqParamEstim
```

The key function is `DiffEqParamEstim.build_loss_objective()`, which builds a loss (objective) function for the problem against the data. Then we can use optimization packages to solve the problem.

### Estimate a single parameter from the data and the ODE model

Let's optimize the parameters of the Lotka-Volterra equation.
===#

using DifferentialEquations
using Plots
using DiffEqParamEstim
using ForwardDiff
using Optimization
using OptimizationOptimJL

## Example model
function lotka_volterra!(du, u, p, t)
    du[1] = dx = p[1] * u[1] - u[1] * u[2]
    du[2] = dy = -3 * u[2] + u[1] * u[2]
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5] ## The true parameter value
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Create a sample dataset with some noise.
ts = range(tspan[begin], tspan[end], 200)
data = [sol.(ts, idxs=1) sol.(ts, idxs=2)] .* (1 .+ 0.03 .* randn(length(ts), 2))

# Plotting the sample dataset and the true solution.
plot(sol)
scatter!(ts, data, label=["u1 data" "u2 data"])

#===
`DiffEqParamEstim.build_loss_objective()` builds a loss function for the ODE problem for the data.

We will minimize the mean squared error using `L2Loss()`.

Note that
- the data should be transposed.
- Uses `AutoForwardDiff()` as the automatic differentiation (AD) method since the number of parameters plus states is small (<100). For larger problems, one can use `Optimization.AutoZygote()`.

===#
alg = Tsit5()

cost_function = build_loss_objective(
    prob, alg,
    L2Loss(collect(ts), transpose(data)),
    Optimization.AutoForwardDiff(),
    maxiters=10000, verbose=false
)

plot(
    cost_function, 0.0, 10.0,
    linewidth=3, label=false, yscale=:log10,
    xaxis="Parameter", yaxis="Cost", title="1-Parameter Cost Function"
)

# There is a dip (minimum) in the cost function at the true parameter value (1.5). We can use an optimizer, e.g., `Optimization.jl`, to find the parameter value that minimizes the cost. (1.5 in this case)
optprob = Optimization.OptimizationProblem(cost_function, [1.42])
optsol = solve(optprob, BFGS())

# The fitting result:
newprob = remake(prob, p=optsol.u)
newsol = solve(newprob, Tsit5())
plot(sol)
plot!(newsol)

# ### Estimate multiple parameters
# Let's use the Lotka-Volterra (Fox-rabbit) equations with all 4 parameters free.

function f2(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]  ## True parameters
alg = Tsit5()
prob = ODEProblem(f2, u0, tspan, p)
sol = solve(prob, alg)

#---
ts = range(tspan[begin], tspan[end], 200)
data = [sol.(ts, idxs=1) sol.(ts, idxs=2)] .* (1 .+ 0.01 .* randn(length(ts), 2))

# Then we can find multiple parameters at once using the same steps. True parameters are `[1.5, 1.0, 3.0, 1.0]`.
cost_function = build_loss_objective(
    prob, alg, L2Loss(collect(ts), transpose(data)),
    Optimization.AutoForwardDiff(),
    maxiters=10000, verbose=false
)
optprob = Optimization.OptimizationProblem(cost_function, [1.3, 0.8, 2.8, 1.2])
result_bfgs = solve(optprob, BFGS())
