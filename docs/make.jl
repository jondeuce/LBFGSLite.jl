using LBFGSLite
using Documenter

DocMeta.setdocmeta!(LBFGSLite, :DocTestSetup, :(using LBFGSLite); recursive=true)

makedocs(;
    modules=[LBFGSLite],
    authors="Jonathan Doucette <jdoucette@physics.ubc.ca> and contributors",
    sitename="LBFGSLite.jl",
    format=Documenter.HTML(;
        canonical="https://jondeuce.github.io/LBFGSLite.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jondeuce/LBFGSLite.jl",
    devbranch="master",
)
