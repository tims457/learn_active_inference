# Activate local environment, see `Project.toml`
import Pkg; Pkg.activate("."); Pkg.instantiate();

using RxInfer, Random, Plots, StableRNGs, LinearAlgebra, StatsPlots, LaTeXStrings, DataFrames, CSV, GLM
##

function generate_data(a, b, v, nr_samples; rng=StableRNG(1234))
    x = float.(collect(1:nr_samples))
    y = a .* x .+ b .+ randn(rng, nr_samples) .* sqrt(v)
    return x, y
end;

##
x_data, y_data = generate_data(0.5, 25.0, 1.0, 250)

scatter(x_data, y_data, title = "Dataset (City road)", legend=false)
xlabel!("Speed")
ylabel!("Fuel consumption")

##

@model function linear_regression(nr_samples)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)
    
    x = datavar(Float64, nr_samples)
    y = datavar(Float64, nr_samples)
    
    y .~ Normal(mean = a .* x .+ b, variance = 1.0)
end

##
results = infer(
    model        = linear_regression(length(x_data)), 
    data         = (y = y_data, x = x_data), 
    initmessages = (b = NormalMeanVariance(0.0, 100.0), ), 
    returnvars   = (a = KeepLast(), b = KeepLast()),
    iterations   = 200,
    free_energy  = true
)

##
pra = plot(range(-3, 3, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 1.0), x), title=L"Prior for $a$ parameter", fillalpha=0.3, fillrange = 0, label=L"$p(a)$", c=1,)
pra = vline!(pra, [ 0.5 ], label=L"True $a$", c = 3)
psa = plot(range(0.45, 0.55, length = 1000), (x) -> pdf(results.posteriors[:a], x), title=L"Posterior for $a$ parameter", fillalpha=0.3, fillrange = 0, label=L"$p(a\mid y)$", c=2,)
psa = vline!(psa, [ 0.5 ], label=L"True $a$", c = 3)

plot(pra, psa, size = (1000, 200), xlabel=L"$a$", ylabel=L"$p(a)$", ylims=[0,Inf])

##
prb = plot(range(-40, 40, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 100.0), x), title=L"Prior for $b$ parameter", fillalpha=0.3, fillrange = 0, label=L"p(b)", c=1, legend = :topleft)
prb = vline!(prb, [ 25 ], label=L"True $b$", c = 3)
psb = plot(range(23, 28, length = 1000), (x) -> pdf(results.posteriors[:b], x), title=L"Posterior for $b$ parameter", fillalpha=0.3, fillrange = 0, label=L"p(b\mid y)", c=2, legend = :topleft)
psb = vline!(psb, [ 25 ], label=L"True $b$", c = 3)

plot(prb, psb, size = (1000, 200), xlabel=L"$b$", ylabel=L"$p(b)$", ylims=[0, Inf])

##


a = results.posteriors[:a]
b = results.posteriors[:b]

println("Real a: ", 0.5, " | Estimated a: ", mean_var(a), " | Error: ", abs(mean(a) - 0.5))
println("Real b: ", 25.0, " | Estimated b: ", mean_var(b), " | Error: ", abs(mean(b) - 25.0))

##
# drop first iteration, which is influenced by the `initmessages`
plot(2:length(results.free_energy), results.free_energy[2:end], title="Free energy", xlabel="Iteration", ylabel="Free energy [nats]", legend=false)

## ===============================================================
# Univariate regression with uknown noise

# In Bayesian statistics, the Inverse Gamma (IG) distribution is prominently used as a conjugate prior for the variance parameter of a normal distribution when both the mean and variance are unknown. This usage is particularly advantageous because it simplifies the Bayesian updating process, allowing for analytical solutions to posterior distributions, which are essential for Bayesian inference.

# The Inverse Gamma distribution is used to estimate the variance of a normal distribution in Bayesian statistics because it is a conjugate prior for the variance parameter. This means that when the Inverse Gamma distribution is used as the prior distribution for the variance, and the likelihood of the data is modeled as a normal distribution, the resulting posterior distribution for the variance will also be an Inverse Gamma distribution. This conjugacy property is highly beneficial because it allows for the posterior distributions to be derived in a closed-form, which simplifies the process of Bayesian updating.

@model function linear_regression_unknown_noise(nr_samples)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)
    s ~ InverseGamma(1.0, 1.0)
    
    x = datavar(Float64, nr_samples)
    y = datavar(Float64, nr_samples)
    
    y .~ Normal(mean = a .* x .+ b, variance = s)
end

x_data_un, y_data_un = generate_data(0.5, 25.0, 400.0, 250)

scatter(x_data_un, y_data_un, title = "Dateset with unknown noise (mountain road)", legend=false)
xlabel!("Speed")
ylabel!("Fuel consumption")

##
results_unknown_noise = infer(
    model           = linear_regression_unknown_noise(length(x_data_un)), 
    data            = (y = y_data_un, x = x_data_un), 
    initmessages    = (b = NormalMeanVariance(0.0, 100.0), ), 
    returnvars      = (a = KeepLast(), b = KeepLast(), s = KeepLast()), 
    iterations      = 20,
    constraints     = MeanField(),
    initmarginals   = (s = vague(InverseGamma), ),
    free_energy     = true
)
@show results_unknown_noise

plot(results_unknown_noise.free_energy, title="Free energy", xlabel="Iteration", ylabel="Free energy [nats]", legend=false)

##

pra = plot(range(-3, 3, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 1.0), x), title=L"Prior for $a$ parameter", fillalpha=0.3, fillrange = 0, label=L"$p(a)$", c=1,)
pra = vline!(pra, [ 0.5 ], label=L"True $a$", c = 3)
psa = plot(range(0.45, 0.55, length = 1000), (x) -> pdf(results_unknown_noise.posteriors[:a], x), title=L"Posterior for $a$ parameter", fillalpha=0.3, fillrange = 0, label=L"$q(a)$", c=2,)
psa = vline!(psa, [ 0.5 ], label=L"True $a$", c = 3)

plot(pra, psa, size = (1000, 200), xlabel=L"$a$", ylabel=L"$p(a)$", ylims=[0, Inf])

##

prb = plot(range(-40, 40, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 100.0), x), title=L"Prior for $b$ parameter", fillalpha=0.3, fillrange = 0, label=L"$p(b)$", c=1, legend = :topleft)
prb = vline!(prb, [ 25.0 ], label=L"True $b$", c = 3)
psb = plot(range(23, 28, length = 1000), (x) -> pdf(results_unknown_noise.posteriors[:b], x), title=L"Posterior for $b$ parameter", fillalpha=0.3, fillrange = 0, label=L"$q(b)$", c=2, legend = :topleft)
psb = vline!(psb, [ 25.0 ], label=L"True $b$", c = 3)

plot(prb, psb, size = (1000, 200), xlabel=L"$b$", ylabel=L"$p(b)$", ylims=[0, Inf])

##

prb = plot(range(0.001, 400, length = 1000), (x) -> pdf(InverseGamma(1.0, 1.0), x), title=L"Prior for $s$ parameter", fillalpha=0.3, fillrange = 0, label=L"$p(s)$", c=1, legend = :topleft)
prb = vline!(prb, [ 200 ], label=L"True $s$", c = 3)
psb = plot(range(0.001, 400, length = 1000), (x) -> pdf(results_unknown_noise.posteriors[:s], x), title=L"Posterior for $s$ parameter", fillalpha=0.3, fillrange = 0, label=L"$q(s)$", c=2, legend = :topleft)
psb = vline!(psb, [ 200 ], label=L"True $s$", c = 3)

plot(prb, psb, size = (1000, 200), xlabel=L"$s$", ylabel=L"$p(s)$", ylims=[0, Inf])


##

as = rand(results_unknown_noise.posteriors[:a], 100)
bs = rand(results_unknown_noise.posteriors[:b], 100)
p = scatter(x_data_un, y_data_un, title = "Linear regression with more noise", legend=false)
xlabel!("Speed")
ylabel!("Fuel consumption")
for (a, b) in zip(as, bs)
    global p = plot!(p, x_data_un, a .* x_data_un .+ b, alpha = 0.05, color = :red)
end

plot(p, size = (900, 400))


## ===============================================================
# Multivariate regression

@model function linear_regression_multivariate(dim, nr_samples)
    a ~ MvNormal(mean = zeros(dim), covariance = 100 * diageye(dim))
    b ~ MvNormal(mean = ones(dim), covariance = 100 * diageye(dim))
    W ~ InverseWishart(dim + 2, 100 * diageye(dim))

    # Here is a small trick to make the example work
    # We treat the `x` vector as a Diagonal matrix such that we can easily multiply it with `a`
    x = datavar(Diagonal{Float64, Vector{Float64}}, nr_samples)
    y = datavar(Vector{Float64}, nr_samples)
    z = randomvar(nr_samples)

    z .~ x .* a .+ b
    y .~ MvNormal(mean = z, covariance = W)

end

dim_mv = 6
nr_samples_mv = 50
rng_mv = StableRNG(42)
a_mv = randn(rng_mv, dim_mv)
b_mv = 10 * randn(rng_mv, dim_mv)
v_mv = 100 * rand(rng_mv, dim_mv)

x_data_mv, y_data_mv = collect(zip(generate_data.(a_mv, b_mv, v_mv, nr_samples_mv)...));

p = plot(title = "Multivariate linear regression", legend = :topleft)

plt = palette(:tab10)

data_set_label = [""]

for k in 1:dim_mv
    global p = scatter!(p, x_data_mv[k], y_data_mv[k], label = "Measurement #$k", ms = 2, color = plt[k])
end
xlabel!(L"$x$")
ylabel!(L"$y$")
p
##

x_data_mv_processed = map(i -> Diagonal([getindex.(x_data_mv, i)...]), 1:nr_samples_mv)
y_data_mv_processed = map(i -> [getindex.(y_data_mv, i)...], 1:nr_samples_mv);


results_mv = infer(
    model           = linear_regression_multivariate(dim_mv, nr_samples_mv),
    data            = (y = y_data_mv_processed, x = x_data_mv_processed),
    initmarginals   = (W = InverseWishart(dim_mv + 2, 10 * diageye(dim_mv)), ),
    initmessages    = (b = MvNormalMeanCovariance(ones(dim_mv), 10 * diageye(dim_mv)), ),
    returnvars      = (a = KeepLast(), b = KeepLast(), W = KeepLast()),
    free_energy     = true,
    iterations      = 50,
    constraints     = MeanField()
)

p = plot(title = "Multivariate linear regression", legend = :topleft, xlabel=L"$x$", ylabel=L"$y$")

# how many lines to plot
r = 50

i_a = collect.(eachcol(rand(results_mv.posteriors[:a], r)))
i_b = collect.(eachcol(rand(results_mv.posteriors[:b], r)))

plt = palette(:tab10)

for k in 1:dim_mv
    x_mv_k = x_data_mv[k]
    y_mv_k = y_data_mv[k]

    for i in 1:r
        global p = plot!(p, x_mv_k, x_mv_k .* i_a[i][k] .+ i_b[i][k], label = nothing, alpha = 0.05, color = plt[k])
    end

    global p = scatter!(p, x_mv_k, y_mv_k, label = "Measurement #$k", ms = 2, color = plt[k])
end

# truncate the init step
f = plot(results_mv.free_energy[2:end], title ="Bethe free energy convergence", label = nothing, xlabel = "Iteration", ylabel = "Bethe free energy [nats]") 

plot(p, f, size = (1000, 400))


##
i_a_mv = results_mv.posteriors[:a]

ps_a = []

for k in 1:dim_mv
    
    local _p = plot(title = L"Estimated $a_{%$k}$", xlabel=L"$a_{%$k}$", ylabel=L"$p(a_{%$k})$", xlims = (-1.5,1.5), xticks=[-1.5, 0, 1.5], ylims=[0, Inf], size = (1000, 800))

    local m_a_mv_k = mean(i_a_mv)[k]
    local v_a_mv_k = std(i_a_mv)[k, k]
    
    _p = plot!(_p, Normal(m_a_mv_k, v_a_mv_k), fillalpha=0.3, fillrange = 0, label=L"$q(a_{%$k})$", c=2,)
    _p = vline!(_p, [ a_mv[k] ], label=L"True $a_{%$k}$", c = 3)
           
    push!(ps_a, _p)
end

plot(ps_a...)

##
scatter(1:dim_mv, v_mv, ylims=(0, 100), label=L"True $s_d$")
scatter!(1:dim_mv, diag(mean(results_mv.posteriors[:W])); yerror=sqrt.(diag(var(results_mv.posteriors[:W]))), label=L"$\mathrm{E}[s_d] \pm \sigma$")
plot!(; xlabel=L"Dimension $d$", ylabel="Variance", title="Estimated variance of the noise")


## ===============================================================
# Hierarchical Bayesian linear regression
# data from https://gist.githubusercontent.com/ucals/2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/43034c39052dcf97d4b894d2ec1bc3f90f3623d9/osic_pulmonary_fibrosis.csv

dataset = CSV.read("pulmonary_fibrosis.csv", DataFrame)
@show describe(dataset)

@show first(dataset, 5)

##

patientinfo(dataset, patient_id) = filter(:Patient => ==(patient_id), dataset)

function patientchart(dataset, patient_id; line_kws = true)
    info = patientinfo(dataset, patient_id)
    x = info[!, "Weeks"]
    y = info[!, "FVC"]

    p = plot(tickfontsize = 10, margin = 1Plots.cm, size = (400, 400), titlefontsize = 11)
    p = scatter!(p, x, y, title = patient_id, legend = false, xlabel = "Weeks", ylabel = "FVC")
    
    if line_kws
        # Use the `GLM.jl` package to estimate linear regression
        linearFormulae = @formula(FVC ~ Weeks)
        linearRegressor = lm(linearFormulae, patientinfo(dataset, patient_id))
        linearPredicted = predict(linearRegressor)
        p = plot!(p, x, linearPredicted, color = :red, lw = 3)
    end

    return p
end

p1 = patientchart(dataset, "ID00007637202177411956430")
p2 = patientchart(dataset, "ID00009637202177434476278")
p3 = patientchart(dataset, "ID00010637202177584971671")

plot(p1, p2, p3, layout = @layout([ a b c ]), size = (1200, 400))


##

@model function partially_pooled(patient_codes, weeks)
    μ_α ~ Normal(mean = 0.0, var = 250000.0) # Prior for the mean of α (intercept)
    μ_β ~ Normal(mean = 0.0, var = 9.0)      # Prior for the mean of β (slope)
    σ_α ~ Gamma(shape = 1.75, scale = 45.54) # Prior for the precision of α (intercept)
    σ_β ~ Gamma(shape = 1.75, scale = 1.36)  # Prior for the precision of β (slope)

    n_codes = length(patient_codes)            # Total number of data points
    n_patients = length(unique(patient_codes)) # Number of unique patients in the data

    α = randomvar(n_patients)                # Individual intercepts for each patient
    β = randomvar(n_patients)                # Individual slopes for each patient

    for i in 1:n_patients
        α[i] ~ Normal(mean = μ_α, precision = σ_α) # Sample the intercept α from a Normal distribution
        β[i] ~ Normal(mean = μ_β, precision = σ_β) # Sample the slope β from a Normal distribution
    end

    σ ~ Gamma(shape = 1.75, scale = 45.54)   # Prior for the standard deviation of the error term
    
    FVC_est = randomvar(n_codes)
    data = datavar(Int, n_codes) # Observed FVC measurements

    for i in 1:n_codes
        FVC_est[i] ~ α[patient_codes[i]] + β[patient_codes[i]] * weeks[i] # FVC estimation using patient-specific α and β
        data[i] ~ Normal(mean = FVC_est[i], precision = σ)              # Likelihood of the observed FVC data
    end
end


@constraints function partially_pooled_constraints()
    # Assume that `μ_α`, `σ_α`, `μ_β`, `σ_β` and `σ` are jointly independent
    q(μ_α, σ_α, μ_β, σ_β, σ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
    # Assume that `μ_α`, `σ_α`, `α` are jointly independent
    q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
    # Assume that `μ_β`, `σ_β`, `β` are jointly independent
    q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
    # Assume that `FVC_est`, `σ` are jointly independent
    q(FVC_est, σ) = q(FVC_est)q(σ) 
end

##
patient_ids          = dataset[!, "Patient"] # get the column of all patients
patient_code_encoder = Dict(map(((id, patient), ) -> patient => id, enumerate(unique(patient_ids))));
patient_code_column  = map(patient -> patient_code_encoder[patient], patient_ids)

dataset[!, :PatientCode] = patient_code_column

first(patient_code_encoder, 5)

##

function partially_pooled_inference(dataset)

    patient_codes = values(dataset[!, "PatientCode"])
    weeks = values(dataset[!, "Weeks"])
    FVC_obs = values(dataset[!, "FVC"]);

    results = infer(
        model = partially_pooled(patient_codes, weeks),
        data = (data = FVC_obs, ),
        options = (limit_stack_depth = 500, ),
        constraints = partially_pooled_constraints(),
        initmessages = (
            α = vague(NormalMeanVariance),
            β = vague(NormalMeanVariance),
        ),
        initmarginals = (
            α = vague(NormalMeanVariance),
            β = vague(NormalMeanVariance),
            σ = vague(Gamma),
            σ_α = vague(Gamma),
            σ_β = vague(Gamma),
        ),
        returnvars = KeepLast(),
        iterations = 100
    )
    
end

partially_pooled_inference_results = partially_pooled_inference(dataset)

##

# Convert to `Normal` since it supports easy plotting with `StatsPlots`
let 
    local μ_α = Normal(mean_std(partially_pooled_inference_results.posteriors[:μ_α])...)
    local μ_β = Normal(mean_std(partially_pooled_inference_results.posteriors[:μ_β])...)
    local α = map(d -> Normal(mean_std(d)...), partially_pooled_inference_results.posteriors[:α])
    local β = map(d -> Normal(mean_std(d)...), partially_pooled_inference_results.posteriors[:β])
    
    local p1 = plot(μ_α, title = "q(μ_α)", fill = 0, fillalpha = 0.2, label = false)
    local p2 = plot(μ_β, title = "q(μ_β)", fill = 0, fillalpha = 0.2, label = false)
    
    local p3 = plot(title = "q(α)...", legend = false)
    local p4 = plot(title = "q(β)...", legend = false)
    
    foreach(d -> plot!(p3, d), α) # Add each individual `α` on plot `p3`
    foreach(d -> plot!(p4, d), β) # Add each individual `β` on plot `p4`
    
    plot(p1, p2, p3, p4, size = (1200, 400), layout = @layout([ a b; c d ]))
end

##
function patientchart_bayesian(results, dataset, encoder, patient_id; kwargs...)
    info            = patientinfo(dataset, patient_id)
    patient_code_id = encoder[patient_id]

    patient_α = results.posteriors[:α][patient_code_id]
    patient_β = results.posteriors[:β][patient_code_id]

    estimated_σ = inv(mean(results.posteriors[:σ]))
    
    predict_weeks = range(-12, 134)

    predicted = map(predict_weeks) do week
        pm = mean(patient_α) + mean(patient_β) * week
        pv = var(patient_α) + var(patient_β) * week ^ 2 + estimated_σ
        return pm, sqrt(pv)
    end
    
    p = patientchart(dataset, patient_id; kwargs...)
    
    return plot!(p, predict_weeks, getindex.(predicted, 1), ribbon = getindex.(predicted, 2), color = :orange)
end


p1 = patientchart_bayesian(partially_pooled_inference_results, dataset, patient_code_encoder, "ID00007637202177411956430")
p2 = patientchart_bayesian(partially_pooled_inference_results, dataset, patient_code_encoder, "ID00009637202177434476278")
p3 = patientchart_bayesian(partially_pooled_inference_results, dataset, patient_code_encoder, "ID00011637202177653955184")

plot(p1, p2, p3, layout = @layout([ a b c ]), size = (1200, 400))





##

















































