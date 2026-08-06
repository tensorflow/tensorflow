"""Lists of huge HLO files for CPU benchmarks."""

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

visibility(CPU_BENCHMARKS_VISIBILITY)

HUGE_HLO_FILES = [
    # go/keep-sorted start
    "gemma3_4b_text_keras_jax_batch1_in8_out100.hlo",
    "gemma4_2b_bf16.hlo",
    # go/keep-sorted end
]
