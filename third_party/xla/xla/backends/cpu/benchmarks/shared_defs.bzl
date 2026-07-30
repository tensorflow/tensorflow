"""Package visibility groups for CPU benchmark HLO file lists and build rules."""

CPU_BENCHMARKS_VISIBILITY = [
    # copybara_removed internal path
    "//xla:__subpackages__",
]

visibility(CPU_BENCHMARKS_VISIBILITY)
