using Documenter
using MLJWrappers

const REPO="github.com/JuliaAI/MLJWrappers.jl"

makedocs(
    modules=[
        MLJWrappers,
    ],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
        size_threshold=1_000_000,
    ),
    pages=[
        "Home" => "index.md",
    ],
    repo=Remotes.GitHub("JuliaAI", "MLJWrappers.jl"),
    warnonly = [:cross_references, :missing_docs],
    sitename="MLJWrappers.jl",
)

deploydocs(
    ; repo=REPO,
    devbranch="dev",
    push_preview=false,
)
