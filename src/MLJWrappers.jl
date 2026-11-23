"""
    MLJWrappers

Module providing various light-weight model wrappers for
[MLJ](https://juliaai.github.io/MLJ.jl/stable/) models.

- [`Transformer`](@ref): For wrapping supervised models that implement transform to have
  them behave as transformers in pipelines

- [`BinaryThresholdPredictor`](@ref): For wrapping a model making binary, probabilistic
  predictions, to make it predict point predictions, based on a user-specified probability
  threshold.

"""
module MLJWrappers

import MLJModelInterface as MMI
using MLJModelInterface: Probabilistic, Deterministic,
    ProbabilisticUnsupervisedDetector,
    ProbabilisticSupervisedDetector,
    DeterministicUnsupervisedDetector,
    DeterministicSupervisedDetector,
    target_scitype,
    reformat,
    selectrows
using ScientificTypes
using CategoricalDistributions


include("transformer.jl")
include("binary_threshold_predictor.jl")

export Transformer, BinaryThresholdPredictor

end # module
