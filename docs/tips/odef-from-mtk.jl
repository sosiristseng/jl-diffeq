# # Get the ODE function from ModelingToolkit
# `f = ODEFunction(sys)` could be useful in visualizing vector fields.
using ModelingToolkit
using OrdinaryDiffEq
using Plots

@independent_variables t
@variables x(t) RHS(t)
@parameters τ
D = Differential(t)

# Equations in MTK use the tilde character (`~`) as equality.
# Every MTK system requires a name. The `@named` macro simply ensures that the symbolic name matches the name in the REPL.
eqs = [
    RHS  ~ (1 - x)/τ,
    D(x) ~ RHS
]

@mtkbuild fol_separate = ODESystem(eqs, t)
f = ODEFunction(fol_separate)

# f(u, p, t) returns the value of derivatives
f([0.0], [1.0], 0.0)

# If you already have the ODEproblem, the function is `prob.f`.
prob = ODEProblem(fol_separate, [x => 0.0], (0.0, 1.0), [τ => 1.0]);
f = prob.f
f([0.0], [1.0], 0.0)
