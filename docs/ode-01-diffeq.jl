#===
# Ordinary Differential Equations (ODEs)

- [DifferentialEquations.jl docs](https://diffeq.sciml.ai/dev/index.html)

Define a model function representing the right-hand-side (RHS) of the system.
  - Out-of-place form: `f(u, p, t)` where `u` is the state variable(s), `p` is the parameter(s), and `t` is the independent variable (usually time). The output is the right hand side (RHS) of the differential equation system.
  - In-place form: `f!(du, u, p, t)`, where the output is saved to `du`. The rest is the same as the out-of-place form. The in-place function may be faster since it does not allocate a new array in each call.
- Initial conditions (`u0`) for the state variable(s).
- (Optional) parameter(s) `p`.
- Define a problem (e.g. `ODEProblem`) using the model function (`f`), initial condition(s) (`u0`), simulation time span (`tspan == (tstart, tend)`), and parameter(s) `p`.
- Solve the problem by calling `solve(prob)`.
===#

#===
## Radioactive decay example

The decaying rate of a nuclear isotope is proportional to its concentration (number):

$$
\frac{d}{dt}C(t) = - \lambda C(t)
$$

**State variable(s)**

- $C(t)$: The concentration (number) of a decaying nuclear isotope.

**Parameter(s)**

- $\lambda$: The rate constant of nuclear decay. The half-life: $t_{\frac{1}{2}} = \frac{ln2}{\lambda}$.
===#
using DifferentialEquations ## using OrdinaryDiffEq
using Plots

# The exponential decay ODE model, out-of-place (3-parameter) form
expdecay(u, p, t) = p * u

# Setup ODE problem
p = -1.0 ## Parameter
u0 = 1.0 ## Initial condition
tspan = (0.0, 2.0) ## Simulation start and end time points
prob = ODEProblem(expdecay, u0, tspan, p) ## Define the problem
sol = solve(prob) ## Solve the problem

# Visualize the solution with `Plots.jl`.
plot(sol)

# ## Solution handling
# https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/
# The mostly used feature is `sol(t)`, the state variables at time `t`. `t` could be a scalar or a vector-like sequence. The result state variables are calculated with interpolation.
sol(1.0)

#---
sol(0.0:0.1:2.0)

# `sol.t`: time points by the solver.
sol.t

# `sol.u`: state variables at `sol.t`.
sol.u

#===
## The SIR model

A more complicated example is the SIR model describing infectious disease spreading. There are 3 state variables and 2 parameters.

$$
\begin{align}
\frac{d}{dt}S(t) &= - \beta S(t)I(t)  \\
\frac{d}{dt}I(t) &= \beta S(t)I(t)  - \gamma I(t)  \\
\frac{d}{dt}R(t) &= \gamma I(t)
\end{align}
$$

**State variable(s)**

- $S(t)$ : the fraction of susceptible people
- $I(t)$ : the fraction of infectious people
- $R(t)$ : the fraction of recovered (or removed) people

**Parameter(s)**

- $\beta$ : the rate of infection when susceptible and infectious people meet
- $\gamma$ : the rate of recovery of infectious people
===#
using DifferentialEquations ## using OrdinaryDiffEq
using Plots

# Here we use the in-place form: `f!(du, u, p ,t)` for the SIR model. The output is written to the first argument `du`, without allocating a new array in each function call.
function sir!(du, u, p, t)
    s, i, r = u
    β, γ = p
    v1 = β * s * i
    v2 = γ * i
    du[1] = -v1
    du[2] = v1 - v2
    du[3] = v2
    return nothing
end

# Setup parameters, initial conditions, time span, and the ODE problem.
p = (β=1.0, γ=0.3)
u0 = [0.99, 0.01, 0.00]
tspan = (0.0, 20.0)
prob = ODEProblem(sir!, u0, tspan, p)

# Solve the problem
sol = solve(prob)

# Visualize the solution
plot(sol, labels=["S" "I" "R"], legend=:right)

# `sol[i]`: all components at time step `i`
sol[2]

# `sol[i, j]`: `i`th component at time step `j`
sol[1, 2]

# `sol[i, :]`: the time series for the `i`th component.
sol[1, :]

# `sol(t, idxs=1)`: the 1st element in time point(s) `t` with interpolation. `t` can be a scalar or a vector.
sol(10, idxs=2)

#---
sol(0.0:0.1:20.0, idxs=2)

#===
## Non-autonomous ODEs

In non-autonomous ODEs, term(s) in the right-hand-side (RHS) maybe time-dependent. For example, in this pendulum model has an external, time-dependent, force.

$$
\begin{aligned}
\dot{\theta} &= \omega(t) \\
\dot{\omega} &= -1.5\frac{g}{l}sin(\theta(t)) + \frac{3}{ml^2}M(t)
\end{aligned}
$$

- $\theta$: pendulum angle
- $\omega$: angular rate
- M: time-dependent external torque
- $l$: pendulum length
- $g$: gradational acceleration
===#
using DifferentialEquations ## using OrdinaryDiffEq
using Plots

function pendulum!(du, u, p, t)
    l = 1.0                             ## length [m]
    m = 1.0                             ## mass [kg]
    g = 9.81                            ## gravitational acceleration [m/s²]
    du[1] = u[2]                        ## θ'(t) = ω(t)
    du[2] = -3g / (2l) * sin(u[1]) + 3 / (m * l^2) * p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
    return nothing
end

u0 = [0.01, 0.0]                     ## initial angular deflection [rad] and angular velocity [rad/s]
tspan = (0.0, 10.0)                  ## time interval

M = t -> 0.1 * sin(t)                   ## external torque [Nm] as the parameter for the pendulum model

prob = ODEProblem(pendulum!, u0, tspan, M)
sol = solve(prob)

plot(sol, linewidth=2, xaxis="t", label=["θ [rad]" "ω [rad/s]"])
plot!(M, tspan..., label="Ext. force")

#===
## Linear ODE system

The ODE system could be anything as long as it returns the derivatives of state variables. In this example, the ODE system is described by a matrix differential operator.

$\dot{u} = Au$
===#
using DifferentialEquations ## using OrdinaryDiffEq
using Plots

A = [
    1.0 0 0 -5
    4 -2 4 -3
    -4 0 0 1
    5 -2 2 3
]

u0 = rand(4, 2)

tspan = (0.0, 1.0)
f = (u, p, t) -> A * u
prob = ODEProblem(f, u0, tspan)
sol = solve(prob)
plot(sol)
