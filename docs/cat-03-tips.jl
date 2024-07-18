# # Calalyst tips

# ## Conservation laws
# We can use conservation laws to eliminate some unknown variables.
# For example, in the chemical reaction `A + B <--> C`, given the initial concentrations of A, B, and C, the solver needs to find only one of [A], [B], and [C] instead of all three.

using Catalyst
using ModelingToolkit
using DifferentialEquations
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
