"""Lists of HLO files for end-to-end CPU benchmarks."""

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

visibility(CPU_BENCHMARKS_VISIBILITY)

REGULAR_HLO_FILES = [
    # go/keep-sorted start
    "argsort_axis_1024x512_bf16.hlo",
    "bench_mass_matrix_35.hlo",
    "bench_scan_N2000_M3.hlo",
    "depthwise_conv_3x3_1x256x56x56_bf16.hlo",
    "diffrax.b380012920.hlo",
    "dlrm_keras_jax_f32_test_step.hlo",
    "dlrm_keras_jax_f32_train_step.hlo",
    "dynamic_slice_loop_1x2048x768_bf16.hlo",
    "in_place_dynamic_update_slice_fusion.hlo",
    "jax.b380442861.hlo",
    "jax.issue.33666.linx.frag_0100.module_0005.hlo",
    "jax.issue.33666.linx.frag_0100.module_0009.hlo",
    "jax.issue.33666.linx.frag_0100.module_0019.hlo",
    "jax.issue.33666.linx.frag_0428.module_0002.hlo",
    "jax.issue.33666.linx.frag_0428.module_0004.hlo",
    "jax.issue.33666.linx.frag_0428.module_0009.hlo",
    "jax.issue.33666.linx.slow_0100.module_0007.hlo",
    "jax.issue.33666.linx.slow_0100.module_0013.hlo",
    "jax.issue.33666.linx.slow_0428.module_0003.hlo",
    "jax.issue.33666.linx.slow_0428.module_0006.hlo",
    "layer_norm_1x4096x768_bf16.hlo",
    "mean_axis_1x4096x1024_bf16.hlo",
    "mha_block_1x12x128x64_bf16.hlo",
    "resnet50_keras_jax_f32.hlo",
    "sort_full_1024x4096_bf16.hlo",
    "sum_axis_1x4096x1024_bf16.hlo",
    "topk_logits_k10_1x50000_bf16.hlo",
    "xnn.parallel_dots.optimized.hlo",
    "xnn.sequential_dots.optimized.hlo",
    # go/keep-sorted end
]

SLOW_HLO_FILES = [
    # go/keep-sorted start
    "gemma3_1b_flax_call.hlo",
    "gemma3_1b_flax_sample_loop.hlo",
    "jax.b380427514.dynamic.hlo",
    "jax.b380427514.regular.hlo",
    # go/keep-sorted end
]

MEMORY_INTENSIVE_HLO_FILES = [
    # go/keep-sorted start
    "gemma2_2b_keras_jax.hlo",
    # go/keep-sorted end
]
