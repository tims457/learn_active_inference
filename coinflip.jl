using RxInfer, Distributions, Random

##
rng = MersenneTwister(42)
n = 10
p = 0.75
distribution = Bernoulli(p)

dataset = float.(rand(rng, Bernoulli(p), n))


##
# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)

    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end

    # We return references to our data inputs and θ parameter
    # We will use these references later on during inference step
    return y, θ
end

##
result = infer(
    model = coin_model(length(dataset)),
    data  = (y = dataset, )
)
θestimated = result.posteriors[:θ]
@show θestimated

println("mean: ", mean(θestimated))
println("std:  ", std(θestimated))

## Custom inference
function custom_inference(data)
    n = length(data)

    # `coin_model` function from `@model` macro returns a reference to the model generator object
    # we need to use the `create_model` function to get actual model object
    model, (y, θ) = create_model(coin_model(n))

    # Reference for future posterior marginal
    mθ = nothing

    # `getmarginal` function returns an observable of future posterior marginal updates
    # We use `Rocket.jl` API to subscribe on this observable
    # As soon as posterior marginal update is available we just save it in `mθ`
    subscription = subscribe!(getmarginal(θ), (m) -> mθ = m)

    # `update!` function passes data to our data inputs
    update!(y, data)

    # It is always a good practice to unsubscribe and to
    # free computer resources held by the subscription
    unsubscribe!(subscription)

    # Here we return our resulting posterior marginal
    return mθ
end

θestimated = custom_inference(dataset)
@show θestimated

println("mean: ", mean(θestimated))
println("std:  ", std(θestimated))

## Plotting
using Plots

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)

plot(p1, p2, layout = @layout([ a; b ]))

##

dataset_100   = float.(rand(rng, Bernoulli(p), 100))
dataset_1000  = float.(rand(rng, Bernoulli(p), 1000))
dataset_10000 = float.(rand(rng, Bernoulli(p), 10000))

θestimated_100   = custom_inference(dataset_100)
θestimated_1000  = custom_inference(dataset_1000)
θestimated_10000 = custom_inference(dataset_10000)

p3 = plot(title = "Posterior", legend = :topleft)

p3 = plot!(p3, rθ, (x) -> pdf(θestimated_100, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_100)", c = 4)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_1000, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_1000)", c = 5)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_10000, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_10000)", c = 6)

plot(p1, p3, layout = @layout([ a; b ]))
##
println("mean: ", mean(θestimated_10000))
println("std:  ", std(θestimated_10000))