#===
# Plotting by Plots.jl

===#

using Plots
using Random
using DisplayAs: PNG
Random.seed!(2022)

#---

plot(rand(1:6, 5)) |> PNG

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()
