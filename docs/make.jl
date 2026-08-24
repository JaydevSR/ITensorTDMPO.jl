using Documenter
using TDVPlus

DocMeta.setdocmeta!(TDVPlus, :DocTestSetup, :(using TDVPlus); recursive = true)

makedocs(;
    modules = [TDVPlus],
    authors = "u0174972",
    sitename = "TDVPlus.jl",
    # No GitHub remote is configured for this repository yet, so there is
    # nothing for Documenter to link "edit this page" to. Once a remote
    # exists, drop `remotes = nothing` and either pass `repo` explicitly
    # or let Documenter infer it from `git remote get-url origin`.
    remotes = nothing,
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


# `deploydocs` publishes the built site to the `gh-pages` branch of a GitHub
# remote. No remote is configured for this repository yet — uncomment and
# fill in `repo` once it lives on GitHub, and see the CI workflow at
# `.github/workflows/CI.yml`, which calls `makedocs`/`deploydocs` on push.
#
# deploydocs(; repo = "github.com/<org>/TDVPlus.jl", devbranch = "main")
