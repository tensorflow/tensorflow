"""Lists of HLO files for end-to-end CPU benchmarks."""

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

visibility(CPU_BENCHMARKS_VISIBILITY)

REGULAR_HLO_FILES = [
    # go/keep-sorted start
    "gemma3_1b_flax_sample_loop.hlo",
    # go/keep-sorted end
]

SLOW_HLO_FILES = [
    # go/keep-sorted start
    "gemma3_1b_flax_call.hlo",
    # go/keep-sorted end
]

MEMORY_INTENSIVE_HLO_FILES = [
    # go/keep-sorted start
    "gemma2_2b_keras_jax.hlo",
    # go/keep-sorted end
]
