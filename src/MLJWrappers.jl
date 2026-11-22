"""
    MLJWrappers

Module providing various light-weight model wrappers for
[MLJ](https://juliaai.github.io/MLJ.jl/stable/) models.

- [`Transformer`](@ref): For wrapping supervised models that implement transform to have
  them behave as transformers in pipelines

"""
module MLJWrappers

import MLJModelInterface as MMI
import MLJModelInterface: Probabilistic, Deterministic,
    ProbabilisticUnsupervisedDetector,
    ProbabilisticSupervisedDetector,
    DeterministicUnsupervisedDetector,
    DeterministicSupervisedDetector
using ScientificTypes
using CategoricalDistributions


include("transformer.jl")
include("binary_threshold_predictor.jl")

export Transformer

end # module
