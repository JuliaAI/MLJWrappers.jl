# # TYPE

mutable struct Transformer{M<:MMI.Model} <: MMI.Unsupervised
    model::M
end;


# # LOGGING

const ERR_MODEL_UNSPECIFIED = ErrorException(
"You must specify a suitable supervised model to wrap, as in "*
    "`Transformer(model=...)`. "
)


# # IMPLEMENTATION OF MLJ INTERFACE

# keyword constructor:
function Transformer(; model=nothing)
    isnothing(model) && throw(ERR_MODEL_UNSPECIFIED)
    return Transformer(model)
end

MMI.reformat(transformer::Transformer, args...) = MMI.reformat(transformer.model, args...)
MMI.fit(transformer::Transformer, args...)=
    MMI.fit(transformer.model, args...)
MMI.update(transformer::Transformer, args...) =
    MMI.update(transformer.model, args...)
MMI.fitted_params(transformer::Transformer, fitresult) =
    MMI.fitted_params(transformer.model, fitresult)
MMI.report(transformer::Transformer, dict) = MMI.report(transformer.model, dict)

MMI.transform(transformer::Transformer, fitresult, Xnew) =
    MMI.transform(transformer.model, fitresult, Xnew)

MMI.metadata_pkg(
    Transformer,
    package_name = "MLJWrappers.jl",
    package_uuid = "b5d0f7f3-9870-4c70-ba08-cb780c37e63f",
    package_url = "https://github.com/JuliaAI/Transformer.jl",
    is_pure_julia = true,
    is_wrapper = true
)

MMI.metadata_model(
    Transformer,
    load_path = "Transformer.Transformer",
)

MMI.target_in_fit(::Type{<:Transformer}) = true


# ## Trait forwarding

for trait in [
    :input_scitype,
    :output_scitype,
    :target_scitype,
    :supports_training_losses,
    :reports_feature_importances,
    :supports_weights,
    :supports_class_weights,
    ]

    quote
        MMI.$trait(::Type{<:Transformer{M}}) where M = MMI.$trait(M)
    end |> eval
end


# ## Other method forwarding

MMI.training_losses(transformer::Transformer, report) =
    MMI.training_losses(transformer.model, report)

MMI.feature_importances(transformer::Transformer, args...) =
    MMI.feature_importances(transformer.model, args...)

# Iteration parameter:
prepend(s::Symbol, ::Nothing) = nothing
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])
quote
    MMI.iteration_parameter(::Type{<:Transformer{M}}) where M =
        prepend(:model, MMI.iteration_parameter(M))
end |> eval


# ## DOCSTRING

"""
    Transformer(supervised_model)

Wrap `supervised_model` so that it is treated as a transformer in MLJ pipelines. It is
assumed that `supervised_model isa Supervised` and that `transform` is implemented for the
model type.

For `Supervised` models in an MLJ pipeline, it is the output of `predict` that is
propagated by default to the next model in the pipeline. By wrapping in `Transform`, the
output of `transform` is propagated instead.

The original hyperparameters of `supervised_model` are nested hyperparameters in
`Transformer(supervised_model)`, but in most other respects the latter behaves like
`supervised_model`.

# Example

Below `reducer` is a supervised model implementing `transform` which
selects features using Recursive Feature Elimination. However, in an MLJ pipeline it is
treated as supervised, leading to the error shown.

```julia
using MLJ
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

reducer = RecursiveFeatureElimination(RandomForestClassifier(), n_features=2)
reducer |> KNNClassifier()
# ERROR: ArgumentError: More than one supervised model in a pipeline is not permitted
```

The following, however, works as expected, passing the reduced training features to the
K-nearest neighbor classifier, when `pipe` is trained.

```
pipe = Transformer(reducer) |> KNNClassifier()
```

"""
Transformer
