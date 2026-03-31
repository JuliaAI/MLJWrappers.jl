using Test
using MLJWrappers
using MLJBase
using MLJDecisionTreeInterface
using FeatureSelection
using StableRNGs

model = RandomForestClassifier(rng=123)
transformer = Transformer(deepcopy(model))

@test transformer isa Unsupervised

# pass-through traits:
@test all([
    input_scitype,
    output_scitype,
    target_scitype,
    supports_training_losses,
    reports_feature_importances,
    supports_weights,
    supports_class_weights,
]) do trait
    trait(transformer) == trait(model)
end

@test iteration_parameter(transformer) == :(model.n_trees)
@test is_wrapper(transformer)

# synthesize some data for training `model` and its wrapped version:
rng = StableRNGs.StableRNG(123)
x1 = rand(rng, 20)
x2 = rand(rng, 20) # dominant feature
x3 = rand(rng, 20)
X = (; x1, x2, x3)
y = coerce(x2 .> 0.5, OrderedFactor);

# train and update atomic model:
data = MLJBase.reformat(model, X, y)
fitresult, cache, rprt = MLJBase.fit(model, 0, data...)
model.n_trees += 10
fitresult, _, rprt = MLJBase.update(model, 0, fitresult, cache, data...)

# train and update wrapped model:
data2 = MLJBase.reformat(transformer, X, y)
fitresult2, cache, rprt2 = MLJBase.fit(transformer, 0, data2...)
transformer.model.n_trees += 10
fitresult2, _, rprt2 = MLJBase.update(transformer, 0, fitresult2, cache, data2...)

# compare fitted_params:
@test fitted_params(transformer, fitresult2)[1].trees[1].featval ==
    fitted_params(model, fitresult)[1].trees[1].featval

# compare reports:
@test rprt2 == rprt
@test MLJBase.MLJModelInterface.report(transformer, Dict(:fit => rprt2)) ==
    MLJBase.MLJModelInterface.report(model, Dict(:fit => rprt))

# compare feature importances:
importances = feature_importances(model, fitresult, rprt)
importances2 = feature_importances(transformer, fitresult2, rprt2)
@test importances2 == importances

# not implemented by atomic model, so returning `nothing` here:
@test training_losses(transformer, rprt2) == training_losses(model, rprt)

# The `reducer` below is a supervised model implementing `transform`. For the above data,
# `reducer` selects the sole feature :x2 (wrapped as a table).
reducer = RecursiveFeatureElimination(model; n_features=1)
@assert reducer isa Supervised
pipe = Transformer(reducer) |> MLJBase.matrix |> vec
mach = machine(pipe, X, y)
fit!(mach, verbosity=0)
@test transform(mach, X) == x2

true
