#===

# Symbolic calculations in Julia

`Symbolics.jl` is a computer Algebra System (CAS) for Julia. The symbols are number-like and follow Julia semantics so we can put them into a regular function to get a symbolic counterpart.

Source:

- [Simulating Big Models in Julia with ModelingToolkit @ JuliaCon 2021 Workshop](https://youtu.be/HEVOgSLBzWA).
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) Github repo and its [docs](https://symbolics.juliasymbolics.org/dev/).

## Caveats about Symbolics.jl

1. `Symbolics.jl` can only handle *traceble*, *quasi-static* expressions. However, some expressions are not quasi-static e.g. factorial. The number of operations depends on the input value.

> Use `@register` to make it a primitive function.

2. Some code paths is *untraceable*, such as conditional statements: `if`...`else`...`end`.

> You can use `ifelse(cond, ex1, ex2)` to make it traceable.

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

#---
@variables x y
x^2 + y^2

# You can use Latexify to see the LaTeX code.
A = [x^2 + y 0 2x
     0       0 2y
     y^2 + x 0 0]

latexify(A)

# Derivative: `Symbolics.derivative(expr, variable)`
Symbolics.derivative(x^2 + y^2, x)

# Gradient: `Symbolics.gradient(expr, [variables])`
Symbolics.gradient(x^2 + y^2, [x, y])

# Jacobian: `Symbolics.jacobian([exprs], [variables])`
Symbolics.jacobian([x^2 + y^2; y^2], [x, y])

# Substitute: `Symbolics.substitute(expr, mapping)`
Symbolics.substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>y^2))

#---
Symbolics.substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>1.0))

# Simplify: `Symbolics.simplify(expr)`
Symbolics.simplify(sin(x)^2 + 2 + cos(x)^2)

# This expresseion gets automatically simplified because it's always true
2x - x

#---
ex = x^2 + y^2 + sin(x)
isequal(2ex, ex + ex)

#---
ex / ex

# Symblic integration: use [SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl). The [Youtube video](https://youtu.be/L47k2zjPU9s) by `doggo dot jl` gives a concise example.

#===
## Custom functions used in Symbolics

With `@register`, the `rand()` in `foo()` will be evaulated as-is and **will not** be expanded by `Symbolics.jl`.

> By default, new functions are traced to the primitives and the symbolic expressions are written on the primitives. However, we can expand the allowed primitives by registering new functions.
===#

# ## More number types
# Complex number
@variables z::Complex

# Array types with subscript
@variables xs[1:18]

#---
xs[1]

# Explicit vector form
collect(xs)

#===

## Example: Rosenbrock function
Wikipedia: https://en.wikipedia.org/wiki/Rosenbrock_function
We use the vector form of Rosenbrock function

===#

rosenbrock(xs) = sum( 1:length(xs)-1) do i
    100*(xs[i+1] - xs[i]^2)^2 + (1 - xs[i])^2
end

# The function is at minimum when xs are all one's
rosenbrock(ones(100))

#---
N = 100
@variables xs[1:N]

# A long list of vector components
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
# Sparse Hessian matrix of the Hessian matrix of the Rosenbrock function w.r.t. to vector components. (298 non-zero values out of 10000 elements)
hes_sp = Symbolics.hessian_sparsity(rxs, xs)

# Visualize the sparse matrix with `Plots.spy()`
using Plots
using DisplayAs: PNG
spy(hes_sp) |> PNG

#===

## Generate functions from symbols

- `build_function(ex, args...)` generates out-of-place (oop) and in-place (ip) function expressions in a pair.
- `build_function(ex, args..., parallel=Symbolics.MultithreadedForm())` generates a parallel algorithm to evaluate the output. See the [example](https://symbolics.juliasymbolics.org/stable/tutorials/auto_parallel/) in the official docs.
- `build_function(ex, args..., target=Symbolics.CTarget())` generates a C function from Julia.

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

# You can save the generated function for later use: `write("function.jl", string(fexprs[2]))`
# Here, `ForwardDiff.jl` checks if our gradient generated from `Symbolics.jl` is correct.

using ForwardDiff
ForwardDiff.gradient(rosenbrock, inxs) ≈ out

# Sparce Hessian matrix, only non-zero expressions are calculated.
hexprs = build_function(hes_sp, xs)
hoop = eval(hexprs[1])
hip = eval(hexprs[2])
hoop(rand(N))
