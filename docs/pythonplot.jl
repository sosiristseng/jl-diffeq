#===
# Plotting with PythonPlot.jl
===#

import PythonPlot as plt
using DisplayAs: PNG
using Random
Random.seed!(2022)

#---

plt.figure()
plt.plot(1:5, rand(1:6, 5))
plt.gcf() |> PNG

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()
