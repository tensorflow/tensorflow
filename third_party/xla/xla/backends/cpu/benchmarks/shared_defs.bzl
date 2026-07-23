"""Package visibility groups for CPU benchmark HLO file lists and build rules."""

visibility(["//xla/backends/cpu/benchmarks/..."])

CPU_BENCHMARKS_VISIBILITY = [
    # copybara_removed internal path
    "//xla:__subpackages__",
]
