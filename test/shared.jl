using Test
using MLJWrappers

@test MLJWrappers.prepend(:n, nothing) == :n
@test MLJWrappers.prepend(:model, :n) == :(model.n)
@test MLJWrappers.prepend(:model, :(pipe.decision_tree.n)) == :(model.pipe.decision_tree.n)
