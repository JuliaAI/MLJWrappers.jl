"""
    MLJWrappers

Module providing various light-weight model wrappers for
[MLJ](https://juliaai.github.io/MLJ.jl/stable/) models.

- [`Transformer`](@ref): For wrapping supervised models that implement transform to have
  them behave as transformers in pipelines

"""
module MLJWrappers 

import MLJModelInterface as MMI

include("transformer.jl")

export Transformer

end # module
