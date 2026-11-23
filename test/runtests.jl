using Test

test_files = [
    "shared.jl",
    "transformer.jl",
    "binary_threshold_predictor.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
