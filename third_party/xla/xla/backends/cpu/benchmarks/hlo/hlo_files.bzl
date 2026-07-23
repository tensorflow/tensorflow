"""Lists of HLO files for end-to-end CPU benchmarks."""

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

visibility(CPU_BENCHMARKS_VISIBILITY)

REGULAR_HLO_FILES = [
    # go/keep-sorted start
    "argsort_axis_1024x512_bf16.hlo",
    "depthwise_conv_3x3_1x256x56x56_bf16.hlo",
    "dynamic_slice_loop_1x2048x768_bf16.hlo",
    "gemma3_1b_flax_sample_loop.hlo",
    "layer_norm_1x4096x768_bf16.hlo",
    "mean_axis_1x4096x1024_bf16.hlo",
    "mha_block_1x12x128x64_bf16.hlo",
    "sort_full_1024x4096_bf16.hlo",
    "sum_axis_1x4096x1024_bf16.hlo",
    "topk_logits_k10_1x50000_bf16.hlo",
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
