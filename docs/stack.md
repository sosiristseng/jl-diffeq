# My Julia stack

- [Gens Julia](https://gensjulia.pages.dev/), my Julia resource list adapted from `Julia.jl`.
- [Julia Package Setup Tutorial](https://bjack205.github.io/tutorial/2021/07/16/julia_package_setup.html)
- [Julia packages (JuliaHub)](https://juliahub.com/ui/Packages)
- [Julia discourse forum](https://discourse.julialang.org/)
- [Solving PDEs in parallel on GPUs with Julia](https://pde-on-gpu.vaw.ethz.ch/) : a Julia course
- [Parallel Computing and Scientific Machine Learning (SciML): Methods and Applications](https://book.sciml.ai/)
- [Modern Julia Workflows](https://modernjuliaworkflows.github.io/)

## Julia stack

- [Agent-based modeling](https://sosiristseng.github.io/jl-abm/)
- [DataFrames tutorial](https://sosiristseng.github.io/jl-dataframes/)
- [Universal differential equations (UDEs)](https://sosiristseng.github.io/jl-ude/)

### Package development

- https://github.com/invenia/PkgTemplates.jl : templates to create new Julia packages.
- https://github.com/timholy/Revise.jl : reducing the need to restart when you make changes to code.
- https://github.com/Roger-luo/FromFile.jl : including other files without duplication.
- https://github.com/JuliaDocs/Documenter.jl : generating package documentation.
- https://github.com/wookay/Jive.jl : unit testing in parallel.

### Publishing code examples

- https://github.com/fredrikekre/Literate.jl : converting _literated_ `jl` files to Markdown (`md`) or Jupyter notebooks (`ipynb`).
- https://github.com/stevengj/NBInclude.jl : converting Jupyter notebooks (`ipynb`) to _literated_ `jl` files by using `nbexport("myfile.jl", "myfile.ipynb")`.
- https://github.com/PumasAI/QuartoNotebookRunner.jl : run [Quarto](https://quarto.org/) notebooks containing Julia code and save the results to Jupyter notebooks.

### Optimization

- https://github.com/baggepinnen/Hyperopt.jl : Hyperparameter optimization for every cost function with multiprocessing and multithreading support.
- https://github.com/SciML/Optimization.jl : A unified interface for various optimizers.
    - `OptimizationBBO` for black-box optimization from https://github.com/robertfeldt/BlackBoxOptim.jl
    - `OptimizationEvolutionary` for genetic algorithm from https://github.com/wildart/Evolutionary.jl
    - `OptimizationOptimJL` for optimization methods from https://github.com/JuliaNLSolvers/Optim.jl
    - `OptimizationMOI` for optimization methods from https://github.com/jump-dev/MathOptInterface.jl
    - etc.

### Curve fitting

- https://github.com/pjabardo/CurveFit.jl
- https://github.com/JuliaNLSolvers/LsqFit.jl
- https://github.com/SciML/Optimization.jl

### Modeling and simulation

- https://github.com/SciML/DifferentialEquations.jl : solving differential equations.
- https://github.com/SciML/ModelingToolkit.jl : a symbolic modeling framework.
  - https://github.com/JuliaSymbolics/Symbolics.jl : a computer algebra system (CAS) for symbolic calculations.
- https://github.com/SciML/Catalyst.jl : a domain-specific language (DSL) for chemical reaction networks.
- https://github.com/JuliaDynamics/Agents.jl : agent-based modeling (ABM).

#### Partial differential equations (PDEs)

- https://github.com/SciML/MethodOfLines.jl : finite difference method (FDM).
- https://github.com/Ferrite-FEM/Ferrite.jl : finite element method (FEM).
- https://github.com/gridap/Gridap.jl : grid-based approximation of PDEs with an expressive API.
- https://github.com/trixi-framework/Trixi.jl : hyperbolic PDE solver.
- https://github.com/j-fu/VoronoiFVM.jl : finite volume method (FVM).

#### Model analysis

- https://github.com/bifurcationkit/BifurcationKit.jl : bifurcation analysis.
- https://github.com/SciML/DiffEqParamEstim.jl

### Probability and Statistics

- https://github.com/JuliaStats/StatsBase.jl : statistics-related functions.
- https://github.com/JuliaStats/GLM.jl : Generalized linear models (GLMs).

### Handy tools

- https://github.com/JuliaMath/Interpolations.jl : continuous interpolation of discrete datasets.
- https://github.com/SciML/DataInterpolations.jl : A library of data interpolation and smoothing functions.
- https://github.com/korsbo/Latexify.jl : convert julia objects to LaTeX equations.

### Arrays

- https://github.com/JuliaArrays/LazyGrids.jl : multi-dimensional grids.
- https://github.com/SciML/LabelledArrays.jl : a label for each element in a array.
- https://github.com/jonniedie/ComponentArrays.jl : arrays with arbitrarily nested named components.

### Concurrency

- https://github.com/JuliaFolds/Folds.jl : a unified interface for sequential, threaded, and distributed fold.
- https://github.com/tkf/ThreadsX.jl : multithreaded base functions.

### Visualization

- [Summary in a discourse thread by Chris Rackauckas](https://discourse.julialang.org/t/comparison-of-plotting-packages/99860/2)

---

- [Plots.jl](https://github.com/JuliaPlots/Plots.jl): powerful and convenient visualization with multiple backends. See also [Plots.jl docs](http://docs.juliaplots.org/latest/)
- [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl): `matplotlib` in Julia. See also [matplotlib docs](https://matplotlib.org/stable/index.html)
- [Makie.jl](https://github.com/JuliaPlots/Makie.jl) see the [Makie.jl docs](https://docs.makie.org/stable/) and [beautiful Makie](https://beautiful.makie.org/).


## Tips

### Get 2D indices from a linear index

Use `CartesianIndices((N, M))`, [source](https://discourse.julialang.org/t/julia-usage-how-to-get-2d-indexes-from-1d-index-when-accessing-a-2d-array/61440).

```julia
x = rand((7, 10))
CI = CartesianIndices((7, 10))

for i in eachindex(x)
    r = CI[i][1]
    c = CI[i][2]
    @assert x[i] == x[r, c]
end
```

### Force display format to PNG

In the VSCode plot panel and https://github.com/fredrikekre/Literate.jl notebooks, PNG images are generally smaller than SVG ones. To force plots to be shown as PNG images, you can use https://github.com/tkf/DisplayAs.jl to show objects in a chosen MIME type.

```julia
using DisplayAs
using Plots

plot(rand(6)) |> DisplayAs.PNG

# If you don't want to add another package dependency, use `display()` directly.

using Plots
PNG(img) = display("image/png", img)
plot(rand(6)) |> PNG
```

### Multiprocessing

[Multi-processing and Distributed Computing](https://docs.julialang.org/en/v1/manual/distributed-computing/) in Julia docs.

#### Activate multiprocessing in Julia

Use `addprocs(N)` in the Julia script

```julia
using Distributed

# Add 3 worker processes
addprocs(3)
```

Or start julia with `-p N`:

```bash
julia -p 3
```

#### Activate environment in worker processes

+ `@everywhere` runs the code in every process.
+ `Base.current_project()` returns the location of `Project.toml`.

```julia
@everywhere begin
    ENV["GKSwstype"] = "100"
    using Pkg
    Pkg.activate(Base.current_project())
end
```
