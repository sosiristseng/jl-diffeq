#===

# Learning Julia Differential Equations

- https://github.com/SciML/DifferentialEquations.jl : high-performance solvers for differential equations.
- https://github.com/SciML/ModelingToolkit.jl : a modeling framework for high-performance symbolic-numeric computation in scientific computing and scientific machine learning
- https://github.com/SciML/Catalyst.jl : a symbolic modeling package for analysis and high performance simulation of chemical reaction networks.

## Tips and tricks

The following section contains tips and tricks in modeling and simulation.

### Smmoth Heaviside step function

A [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) (0 when x < a, 1 when x > a) could be approximated with a steep [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

===#

logistic(x, k=1000) = inv(1 + exp(-k * x))

# The function output switches from zero to one around `x=0`.
using Plots
using DisplayAs: PNG

plot(logistic, -1, 1, label="sigmoid approx.") |> PNG

# ### Smooth single pulse
# A single pulse could be approximated with a product of two logistic functions.

logistic(x, k=1000) = inv(1 + exp(-k * x))
singlepulse(x, t0=0, t1=0.1, k=1000) = logistic(x - t0, k) * logistic(t1 - x, k)

plot(singlepulse, -10, 10, label="sigmoid approx.") |> PNG

# ### Smooth minimal function
# Adapted from this discourse post: https://discourse.julialang.org/t/handling-instability-when-solving-ode-problems/9019/5

function smoothmin(x, k=100)
    -log(1 + exp(-k * x))/k
end

plot(smoothmin, -5, 5, label="Smooth min") |> PNG

# ### Periodic pulses
# From: https://www.noamross.net/2015/11/12/a-smooth-differentiable-pulse-function/

function smoothpulses(t, tstart, tend, period=1, amplitude=period / (tend - tstart), steepness=1000)
    @assert tstart < tend < period
    xi = 3 / 4 - (tend - tstart) / (2 * period)
    p = inv(1 + exp(steepness * (sinpi(2 * ((t - tstart) / period + xi)) - sinpi(2 * xi))))
    return amplitude * p
end

plot(t->smoothpulses(t, 0.2, 0.3, 0.5), 0.0, 2.0, lab="pulses") |> PNG

# ### Avoiding DomainErrors
# Some functions such as `sqrt(x)`, `log(x)`, and `pow(x)` will throw `DomainError` exceptions with negative `x`, interrupting differential equation solvers. One can use the respective functions in https://github.com/JuliaMath/NaNMath.jl, returning `NaN` instead of throwing a `DomainError`. Then, the differential equation solvers will reject the solution and retry with a smaller time step.

import NaNMath as nm
nm.sqrt(-1.0) ## returns NaN

# ## Runtime environment
import InteractiveUtils
InteractiveUtils.versioninfo()

#---
import Pkg
Pkg.status()
