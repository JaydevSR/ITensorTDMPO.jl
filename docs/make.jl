using Documenter
using ITensorTDMPO

DocMeta.setdocmeta!(ITensorTDMPO, :DocTestSetup, :(using ITensorTDMPO); recursive = true)

makedocs(;
    modules = [ITensorTDMPO],
    authors = "JaydevSR",
    sitename = "ITensorTDMPO.jl",
    # Inferred from the `origin` remote now that one is configured — no
    # explicit `remotes`/`repo` override needed for "edit this page" links.
    # Only exported names need a `@docs`/`@autodocs` home; internal helpers
    # (`ChannelSpec`, `cumulative_integral!`, the `Base.empty!` extension)
    # have docstrings for readers of the source, not for the manual.
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Hamiltonian representation" => [
            "Driving channels" => "driving_channels.md",
            "Ramps" => "ramps.md",
        ],
        "Time evolution" => [
            "The time_evolve interface" => "time_evolve.md",
            "Piecewise-constant TDVP" => "piecewise_constant.md",
            "Dyson series" => "dyson.md",
            "Magnus expansion" => "magnus.md",
            "Commutator-free propagator (CFET)" => "cfet.md",
            "Adaptive stepping" => "adaptive.md",
        ],
        "Observables and diagnostics" => "observables.md",
        "Scope and limitations" => "scope.md",
        "API reference" => "api.md",
    ],
)


# Publishes the built site to the `gh-pages` branch of the GitHub remote.
# The `Documentation.yml` workflow calls `makedocs`/`deploydocs` (this
# file) on every push to `main` and on version tags.
deploydocs(; repo = "github.com/JaydevSR/ITensorTDMPO.jl", devbranch = "main")
