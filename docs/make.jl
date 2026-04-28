pushfirst!(LOAD_PATH, normpath(@__DIR__, ".."))

using Documenter
using SymbolicMOR

makedocs(
    sitename = "SymbolicMOR.jl",
    modules = [SymbolicMOR],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "master",
    ),
    checkdocs = :exports,
    warnonly = [:missing_docs],
)
