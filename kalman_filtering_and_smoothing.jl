# Activate local environment, see `Project.toml`
import Pkg; Pkg.activate("."); Pkg.instantiate();
using RxInfer, BenchmarkTools, Random, LinearAlgebra, Plots
##
function generate_data(rng, A, B, Q, P)
    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, Q))
        y[i] = rand(rng, MvNormal(B * x[i], P))
        x_prev = x[i]
    end
    
    return x, y
end

# Seed for reproducibility
seed = 1234

rng = MersenneTwister(1234)

# We will model 2-dimensional observations with rotation matrix `A`
# To avoid clutter we also assume that matrices `A`, `B`, `P` and `Q`
# are known and fixed for all time-steps
θ = π / 35
A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
B = diageye(2)
Q = diageye(2)
P = 25.0 .* diageye(2)

# Number of observations
n = 300;

x, y = generate_data(rng, A, B, Q, P);


px = plot()

px = plot!(px, getindex.(x, 1), label = "Hidden Signal (dim-1)", color = :orange)
px = scatter!(px, getindex.(y, 1), label = false, markersize = 2, color = :orange)
px = plot!(px, getindex.(x, 2), label = "Hidden Signal (dim-2)", color = :green)
px = scatter!(px, getindex.(y, 2), label = false, markersize = 2, color = :green)

plot(px)


##

@model function rotate_ssm(n, x0, A, B, Q, P)
    
    # We create constvar references for better efficiency
    cA = constvar(A)
    cB = constvar(B)
    cQ = constvar(Q)
    cP = constvar(P)
    
    # `x` is a sequence of hidden states
    x = randomvar(n)
    # `y` is a sequence of "clamped" observations
    y = datavar(Vector{Float64}, n)
    
    x_prior ~ MvNormalMeanCovariance(mean(x0), cov(x0))
    x_prev = x_prior
    
    for i in 1:n
        x[i] ~ MvNormalMeanCovariance(cA * x_prev, cQ)
        y[i] ~ MvNormalMeanCovariance(cB * x[i], cP)
        x_prev = x[i]
    end

end

x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2));

# For large number of observations you need to use `limit_stack_depth = 100` option during model creation, e.g. 
# infer(..., options = (limit_stack_depth = 500, ))`
result = infer(
    model = rotate_ssm(length(y), x0, A, B, Q, P), 
    data = (y = y,),
    free_energy = true
);

xmarginals  = result.posteriors[:x]
logevidence = -result.free_energy; # given the analytical solution, free energy will be equal to the negative log evidence


##


px = plot()

px = plot!(px, getindex.(x, 1), label = "Hidden Signal (dim-1)", color = :orange)
px = plot!(px, getindex.(x, 2), label = "Hidden Signal (dim-2)", color = :green)

px = plot!(px, getindex.(mean.(xmarginals), 1), ribbon = getindex.(var.(xmarginals), 1) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-1)", color = :teal)
px = plot!(px, getindex.(mean.(xmarginals), 2), ribbon = getindex.(var.(xmarginals), 2) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-1)", color = :violet)

plot(px)



##

@show logevidence

# @benchmark infer(
#     model = rotate_ssm(length($y), $x0, $A, $B, $Q, $P), 
#     data = (y = $y,)
# )


## ===================================================================
# System Identification Problem

using RxInfer, Distributions, StableRNGs, Plots

function generate_data(f, n; seed = 123, x_i_min = -20.0, w_i_min = 20.0, noise = 20.0, real_x_τ = 0.1, real_w_τ = 1.0)

    rng = StableRNG(seed)

    real_x = Vector{Float64}(undef, n)
    real_w = Vector{Float64}(undef, n)
    real_y = Vector{Float64}(undef, n)

    for i in 1:n
        real_x[i] = rand(rng, Normal(x_i_min, sqrt(1.0 / real_x_τ)))
        real_w[i] = rand(rng, Normal(w_i_min, sqrt(1.0 / real_w_τ)))
        real_y[i] = rand(rng, Normal(f(real_x[i], real_w[i]), sqrt(noise)))

        x_i_min = real_x[i]
        w_i_min = real_w[i]
    end
    
    return real_x, real_w, real_y
end

##

n = 250
real_x, real_w, real_y = generate_data(+, n);

pl = plot(title = "Underlying signals")
pl = plot!(pl, real_x, label = "x")
pl = plot!(pl, real_w, label = "w")

pr = plot(title = "Combined y = x + w")
pr = scatter!(pr, real_y, ms = 3, color = :red, label = "y")

plot(pl, pr, size = (800, 300))


##

@model function identification_problem(f, n, m_x_0, τ_x_0, a_x, b_x, m_w_0, τ_w_0, a_w, b_w, a_y, b_y)
    
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x = randomvar(n)
    w = randomvar(n)
    s = randomvar(n)
    y = datavar(Float64, n)
    
    x_i_min = x0
    w_i_min = w0
    
    for i in 1:n
        x[i] ~ Normal(mean = x_i_min, precision = τ_x)
        w[i] ~ Normal(mean = w_i_min, precision = τ_w)
        s[i] ~ f(x[i], w[i])
        y[i] ~ Normal(mean = s[i], precision = τ_y)
        
        x_i_min = x[i]
        w_i_min = w[i]
    end
    
end


##

constraints = @constraints begin 
    q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
end


##
m_x_0, τ_x_0 = -20.0, 1.0
m_w_0, τ_w_0 = 20.0, 1.0

# We set relatively strong priors for random walk noise components
# and sort of vague prior for the noise of the observations
a_x, b_x = 0.01, 0.01var(real_x)
a_w, b_w = 0.01, 0.01var(real_w)
a_y, b_y = 1.0, 1.0

# We set relatively strong priors for messages
xinit = map(r -> NormalMeanPrecision(r, τ_x_0), reverse(range(-60, -20, length = n)))
winit = map(r -> NormalMeanPrecision(r, τ_w_0), range(20, 60, length = n))

imessages = (x = xinit, w = winit)
imarginals = (τ_x = GammaShapeRate(a_x, b_x), τ_w = GammaShapeRate(a_w, b_w), τ_y = GammaShapeRate(a_y, b_y))

result = infer(
    model = identification_problem(+, n, m_x_0, τ_x_0, a_x, b_x, m_w_0, τ_w_0, a_w, b_w, a_y, b_y),
    data  = (y = real_y,), 
    options = (limit_stack_depth = 500, ), 
    constraints = constraints, 
    initmessages = imessages, 
    initmarginals = imarginals, 
    iterations = 50
)
@show result

τ_x_marginals = result.posteriors[:τ_x]
τ_w_marginals = result.posteriors[:τ_w]
τ_y_marginals = result.posteriors[:τ_y]

smarginals = result.posteriors[:s]
xmarginals = result.posteriors[:x]
wmarginals = result.posteriors[:w];


##
px1 = plot(legend = :bottomleft, title = "Estimated hidden signals")
px2 = plot(legend = :bottomright, title = "Estimated combined signals")

px1 = plot!(px1, real_x, label = "Real hidden X")
px1 = plot!(px1, mean.(xmarginals[end]), ribbon = var.(xmarginals[end]), label = "Estimated X")

px1 = plot!(px1, real_w, label = "Real hidden W")
px1 = plot!(px1, mean.(wmarginals[end]), ribbon = var.(wmarginals[end]), label = "Estimated W")

px2 = scatter!(px2, real_y, label = "Observations", ms = 2, alpha = 0.5, color = :red)
px2 = plot!(px2, mean.(smarginals[end]), ribbon = std.(smarginals[end]), label = "Combined estimated signal", color = :green)

plot(px1, px2, size = (800, 300))

## ===================================================================
# Combination 2: y = min(x, w)


# Smoothed version of `min` without zero-ed derivatives
function smooth_min(x, y)    
    if x < y
        return x + 1e-4 * y
    else
        return y + 1e-4 * x
    end
end

min_meta = @meta begin 
    # In this example we are going to use a simple `Linearization` method
    smooth_min() -> Linearization()
end

n = 200
min_real_x, min_real_w, min_real_y = generate_data(min, n, seed = 1, x_i_min = 0.0, w_i_min = 0.0, noise = 1.0, real_x_τ = 1.0, real_w_τ = 1.0);

pl = plot(title = "Underlying signals")
pl = plot!(pl, min_real_x, label = "x")
pl = plot!(pl, min_real_w, label = "w")

pr = plot(title = "Combined y = min(x, w)")
pr = scatter!(pr, min_real_y, ms = 3, color = :red, label = "y")

plot(pl, pr, size = (800, 300))

##


min_m_x_0, min_τ_x_0 = -1.0, 1.0
min_m_w_0, min_τ_w_0 = 1.0, 1.0

min_a_x, min_b_x = 1.0, 1.0
min_a_w, min_b_w = 1.0, 1.0
min_a_y, min_b_y = 1.0, 1.0

min_imessages = (x = NormalMeanPrecision(min_m_x_0, min_τ_x_0), w = NormalMeanPrecision(min_m_w_0, min_τ_w_0))
min_imarginals = (τ_x = GammaShapeRate(min_a_x, min_b_x), τ_w = GammaShapeRate(min_a_w, min_b_w), τ_y = GammaShapeRate(min_a_y, min_b_y))

min_result = infer(
    model = identification_problem(smooth_min, n, min_m_x_0, min_τ_x_0, min_a_x, min_b_x, min_m_w_0, min_τ_w_0, min_a_w, min_b_w, min_a_y, min_b_y),
    data  = (y = min_real_y,), 
    meta = min_meta,
    options = (limit_stack_depth = 500, ), 
    constraints = constraints, 
    initmessages = min_imessages, 
    initmarginals = min_imarginals, 
    iterations = 100
)

##

min_τ_x_marginals = min_result.posteriors[:τ_x]
min_τ_w_marginals = min_result.posteriors[:τ_w]
min_τ_y_marginals = min_result.posteriors[:τ_y]

min_smarginals = min_result.posteriors[:s]
min_xmarginals = min_result.posteriors[:x]
min_wmarginals = min_result.posteriors[:w]

px1 = plot(legend = :bottomleft, title = "Estimated hidden signals")
px2 = plot(legend = :bottomright, title = "Estimated combined signals")

px1 = plot!(px1, min_real_x, label = "Real hidden X")
px1 = plot!(px1, mean.(min_xmarginals[end]), ribbon = var.(min_xmarginals[end]), label = "Estimated X")

px1 = plot!(px1, min_real_w, label = "Real hidden W")
px1 = plot!(px1, mean.(min_wmarginals[end]), ribbon = var.(min_wmarginals[end]), label = "Estimated W")

px2 = scatter!(px2, min_real_y, label = "Observations", ms = 2, alpha = 0.5, color = :red)
px2 = plot!(px2, mean.(min_smarginals[end]), ribbon = std.(min_smarginals[end]), label = "Combined estimated signal", color = :green)

plot(px1, px2, size = (800, 300))



## ===================================================================
# Online (filtering) identification: y = min(x, w)

@model function rx_identification(f)
    
    # We are going to continuosly update our priors
    # based on new posteriors
    m_x_0 = datavar(Float64) 
    τ_x_0 = datavar(Float64)
    m_w_0 = datavar(Float64) 
    τ_w_0 = datavar(Float64)
    a_x   = datavar(Float64) 
    b_x   = datavar(Float64)
    a_y   = datavar(Float64) 
    b_y   = datavar(Float64)
    a_w   =  datavar(Float64) 
    b_w   = datavar(Float64)
    s     = randomvar()
    y     = datavar(Float64)
    
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x ~ Normal(mean = x0, precision = τ_x)
    w ~ Normal(mean = w0, precision = τ_w)

    s ~ f(x, w)
    y ~ Normal(mean = s, precision = τ_y)
    
end

rx_constraints = @constraints begin 
    q(x0, x, w0, w, τ_x, τ_w, τ_y, s) = q(x0, x)q(w, w0)q(τ_w)q(τ_x)q(s)q(τ_y)
end

autoupdates = @autoupdates begin 
    m_x_0, τ_x_0 = mean_precision(q(x))
    m_w_0, τ_w_0 = mean_precision(q(w))
    a_x = shape(q(τ_x)) 
    b_x = rate(q(τ_x))
    a_y = shape(q(τ_y))
    b_y = rate(q(τ_y))
    a_w = shape(q(τ_w)) 
    b_w = rate(q(τ_w))
end

rx_meta = @meta begin 
    smooth_min() -> Linearization()
end

n = 300
rx_real_x, rx_real_w, rx_real_y = generate_data(min, n, seed = 1, x_i_min = 1.0, w_i_min = -1.0, noise = 1.0, real_x_τ = 1.0, real_w_τ = 1.0);

pl = plot(title = "Underlying signals")
pl = plot!(pl, rx_real_x, label = "x")
pl = plot!(pl, rx_real_w, label = "w")

pr = plot(title = "Combined y = min(x, w)")
pr = scatter!(pr, rx_real_y, ms = 3, color = :red, label = "y")

plot(pl, pr, size = (800, 300))

##

engine = infer(
    model         = rx_identification(smooth_min),
    constraints   = rx_constraints,
    data          = (y = rx_real_y,),
    autoupdates   = autoupdates,
    meta          = rx_meta,
    returnvars    = (:x, :w, :τ_x, :τ_w, :τ_y, :s),
    keephistory   = 1000,
    historyvars   =  KeepLast(),
    initmarginals = (w = NormalMeanVariance(-2.0, 1.0), x = NormalMeanVariance(2.0, 1.0), τ_x = GammaShapeRate(1.0, 1.0), τ_w = GammaShapeRate(1.0, 1.0), τ_y = GammaShapeRate(1.0, 20.0)),
    iterations    = 10,
    free_energy = true, 
    free_energy_diagnostics = nothing,
    autostart     = true,
)


rx_smarginals = engine.history[:s]
rx_xmarginals = engine.history[:x]
rx_wmarginals = engine.history[:w];

##
px1 = plot(legend = :bottomleft, title = "Estimated hidden signals")
px2 = plot(legend = :bottomright, title = "Estimated combined signals")

px1 = plot!(px1, rx_real_x, label = "Real hidden X")
px1 = plot!(px1, mean.(rx_xmarginals), ribbon = var.(rx_xmarginals), label = "Estimated X")

px1 = plot!(px1, rx_real_w, label = "Real hidden W")
px1 = plot!(px1, mean.(rx_wmarginals), ribbon = var.(rx_wmarginals), label = "Estimated W")

px2 = scatter!(px2, rx_real_y, label = "Observations", ms = 2, alpha = 0.5, color = :red)
px2 = plot!(px2, mean.(rx_smarginals), ribbon = std.(rx_smarginals), label = "Combined estimated signal", color = :green)

plot(px1, px2, size = (800, 300))

























