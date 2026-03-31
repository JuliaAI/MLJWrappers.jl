using Documenter
using MLJWrappers

repo = Remotes.GitHub("JuliaAI", "MLJWrappers.jl"),

makedocs(
    ; modules=[MLJWrappers,],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
        size_threshold=1_000_000,
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly = [:cross_references, :missing_docs],
    sitename="MLJWrappers.jl",
    repo,
)

deploydocs(
    ; repo,
    devbranch="dev",
    push_preview=false,
)
