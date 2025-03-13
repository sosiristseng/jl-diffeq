#===
# Symbolic calculations in Julia

`Symbolics.jl` is a computer Algebra System (CAS) for Julia. The symbols are number-like and follow Julia semantics so we can put them into a regular function to get a symbolic counterpart. Symbolics.jl is the backbone of ModelingToolkit.jl. [Comparision to SymPy](https://docs.sciml.ai/Symbolics/stable/comparison/)

Source:

- [Simulating Big Models in Julia with ModelingToolkit @ JuliaCon 2021 Workshop](https://youtu.be/HEVOgSLBzWA).
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) Github repo and its [docs](https://docs.sciml.ai/Symbolics/stable/).

## Caveats about Symbolics.jl

1. `Symbolics.jl` can only handle *traceble*, *quasi-static* expressions. However, some expressions are not quasi-static e.g. factorial. The number of operations depends on the input value.

> Use `@register_symbolic` to stop Symbolics.jl from digging further.

2. Some code paths is *untraceable*, such as conditional statements: `if`...`else`...`end`.

> You can use `ifelse(cond, ex1, ex2)` or `(cond) * (ex1) + (1 - cond) * (ex2)` to make it traceable.

## Basic operations

- `latexify`
- `derivative`
- `gradient`
- `jacobian`
- `substitute`
- `simplify`
===#
using Symbolics
using Latexify

@variables x y
x^2 + y^2

# You can use `Latexify.latexify()` to see the LaTeX code.
A = [
    x^2+y 0 2x
    0 0 2y
    y^2+x 0 0
]

latexify(A)

# Derivative: `Symbolics.derivative(expr, variable)`
Symbolics.derivative(x^2 + y^2, x)

# Gradient: `Symbolics.gradient(expr, [variables])`
Symbolics.gradient(x^2 + y^2, [x, y])

# Jacobian: `Symbolics.jacobian([exprs], [variables])`
Symbolics.jacobian([x^2 + y^2; y^2], [x, y])

# Substitute: `Symbolics.substitute(expr, mapping)`
Symbolics.substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x => y^2))

#---
Symbolics.substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x => 1.0))

# Simplify: `Symbolics.simplify(expr)`
Symbolics.simplify(sin(x)^2 + 2 + cos(x)^2)

# This expression gets automatically simplified because it's always true
2x - x

#---
ex = x^2 + y^2 + sin(x)
isequal(2ex, ex + ex)

#---
isequal(simplify(2ex), simplify(ex + ex))

#---
ex / ex

# Symbolic integration: use [SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl). The [Youtube video](https://youtu.be/L47k2zjPU9s) by `doggo dot jl` gives a concise example.

#===
## Custom functions

https://docs.sciml.ai/Symbolics/stable/manual/functions/

With `@register_symbolic` and `@register_array_symbolic`, functions will be evaluated as-is and **will not** be traced and expanded by `Symbolics.jl`. Useful for `rand()`, data interpolations, etc.
===#

# ## More number types
# Complex number
@variables z::Complex

# Array types with subscript
@variables xs[1:100]

#---
xs[1]

# Explicit vector form
collect(xs)

# Operations on arrays are supported
sum(collect(xs))

#===
## Example: Rosenbrock function

Wikipedia: https://en.wikipedia.org/wiki/Rosenbrock_function

We use the vector form of Rosenbrock function.
===#

rosenbrock(xs) = sum(1:length(xs)-1) do i
    100 * (xs[i+1] - xs[i]^2)^2 + (1 - xs[i])^2
end

# The function is at minimum when xs are all one's
rosenbrock(ones(100))

# Making a 20-element array
N = 20
@variables xs[1:N]

# A full list of vector components
xs = collect(xs)

#---
rxs = rosenbrock(xs)

# Gradient
grad = Symbolics.gradient(rxs, xs)

# Hessian = Jacobian of gradient
hes1 = Symbolics.jacobian(grad, xs)

# call `hessian()` directly
hes2 = Symbolics.hessian(rxs, xs)

#---
isequal(hes1, hes2)

# ### Sparse matrix
# Sparse Hessian matrix of the Hessian matrix of the Rosenbrock function w.r.t. to vector components.
hes_sp = Symbolics.hessian_sparsity(rosenbrock, xs)

# Visualize the sparse matrix with `Plots.spy()`
using Plots
spy(hes_sp)

#===
## Generate functions symbolically

https://docs.sciml.ai/Symbolics/stable/manual/build_function/

- `build_function(ex, args...)` generates out-of-place (oop) and in-place (ip) function expressions in a tuple pair.
- `build_function(ex, args..., parallel=Symbolics.MultithreadedForm())` generates a parallel, multithreaded algorithm.
- `build_function(ex, args..., target=Symbolics.CTarget())` generates a C function from Julia.

For example, we want to build a Julia function from `grad`.
===#
fexprs = build_function(grad, xs);

# Get the Out-of-place `f(input)` version
foop = eval(fexprs[1])

# Get the In-place `f!(out, in)` version
fip = eval(fexprs[2])

#---
inxs = rand(N)
out = similar(inxs)
fip(out, inxs)  ## The inplace version returns nothing. The results are stored in out parameter.

#---
foop(inxs)
isapprox(foop(inxs), out)

#===
To save the generated function for later use:

```julia
write("function.jl", string(fexprs[2]))
```

Load it back

```julia
g = include("function.jl")
```
===#

# Here, `ForwardDiff.jl` checks if our gradient generated from `Symbolics.jl` is correct.
using ForwardDiff: gradient
gradient(rosenbrock, inxs) ≈ out

# Sparse Hessian matrix, only non-zero expressions are calculated.
hexprs = build_function(hes_sp, xs)
hoop = eval(hexprs[1])
hip = eval(hexprs[2])
hoop(rand(N))

#===
## Solve equations symbolically

https://docs.sciml.ai/Symbolics/stable/manual/solver/

Use `symbolic_solve(expr, x)` to get analytic solutions.

For single variable solving, the `Nemo.jl` package is needed; for multiple variable solving, the `Groebner.jl` package is needed.

This example solves the steady-state rate of a enzyme-catalyzed reaction of fumarate hydratase: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-238. At steady-state, the rate of change of each state shoul be zero.
===#
using Symbolics
using Groebner

@variables k1 k2 k3 k4 k5 k6 km1 km2 km3 km4 km5 km6 E1 E2 E3 E4 E5

eqs = let
    v12 = km6 * E1 - k6 * E2
    v13 = k1 * E1 - km1 * E3
    v15 = k3 * E1 - km3 * E5
    v24 = km5 * E2 - k5 * E4
    v34 = k2 * E3 - km2 * E4
    v45 = km4 * E4 - k4 * E5

    dE1 = -v12 - v13 - v15
    dE2 = v12 - v24
    dE3 = v13 - v34
    dE4 = v24 + v34 - v45
    dE5 = v15 + v45

    # Make sure the rates are conserved
    @assert isequal(dE1 + dE2 + dE3 + dE4 + dE5, 0)
    # The following terms should be all zeroes.
    [dE1, dE2, dE3, dE4, E1 + E2 + E3 + E4 + E5 - 1 ]
end

@time sol = Symbolics.symbolic_solve(eqs, [E1, E2, E3, E4, E5])[1]

# The weight of the first enzyme state is
numerator(sol[E1])
