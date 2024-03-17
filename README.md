# LBFGSLite

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/LBFGSLite.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/LBFGSLite.jl/dev/)
[![Build Status](https://github.com/jondeuce/LBFGSLite.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jondeuce/LBFGSLite.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/jondeuce/LBFGSLite.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/LBFGSLite.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Julia port of the [`LBFGS-Lite` C++ header-only library for unconstrained optimization](https://github.com/ZJU-FAST-Lab/LBFGS-Lite).

## Features

From [`LBFGS-Lite`](https://github.com/ZJU-FAST-Lab/LBFGS-Lite):

- This library implements the [Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Method](https://doi.org/10.1007/BF01589116) (L-BFGS).

- A __highly robust line search__ proposed by [Lewis and Overton](https://link.springer.com/article/10.1007/s10107-012-0514-2) is employed.

- Both __smooth__ ([C2](https://en.wikipedia.org/wiki/Smoothness)) and __nonsmooth__ ([C0 but piecewise C2](https://en.wikipedia.org/wiki/Smoothness)) functions are supported.

- __Cautious update__ by [Li and Fukushima](https://epubs.siam.org/doi/pdf/10.1137/S1052623499354242) is employed for __global convergence__ in nonconvex cases.

- __Externally provided maximum step size__ is convenient for functions defined on bounded sets.

## Usage

Minimize the function

$$ f(x) = 1 + \frac{1}{2}(\sin^2 x_1 + \sin^2 x_2) $$

starting from `x₀ = [-2.0, 1.0]` using `optimize`:

```julia
function fg!(x, dx)
    @. dx = sin(x) * cos(x)
    return 1 + sum(abs2 ∘ sin, x) / 2
end

x₀ = Float64[-2.0, 1.0]
params = LBFGSParams{Float64}()
x, fx, st = optimize(fg!, x₀, params)

@assert isapprox(x, [-π, 0.0]; atol = 1e-6)
@assert isapprox(fx, 1.0; atol = 1e-12)
@assert Int(st) == 0
```

Preallocate a workspace and solve the problem in-place using `optimize!`:

```julia
x = copy(x₀) # will be overwritten
params = LBFGSParams{Float64}()
work = LBFGSWorkspace(x, params) # allocate buffers
_, fx, st = optimize!(fg!, x, work, params)

@assert isapprox(x, [-π, 0.0]; atol = 1e-6)
@assert isapprox(fx, 1.0; atol = 1e-12)
@assert Int(st) == 0
```
