load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla:friends",
        "//xla/backends/gpu:__subpackages__",
        "//xla/hlo/tools/hlo_opt:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/tools/hlo_opt:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dot_algorithm_rewriter",
    hdrs = ["dot_algorithm_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dot_algorithm_rewriter"],
)

cc_library(
    name = "algebraic_simplifier",
    hdrs = ["algebraic_simplifier.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:algebraic_simplifier"],
)

cc_library(
    name = "config_aware_cse",
    hdrs = ["config_aware_cse.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:config_aware_cse"],
)

cc_library(
    name = "conv_canonicalizer",
    hdrs = ["conv_canonicalizer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:conv_canonicalizer"],
)

cc_library(
    name = "algorithm_checker",
    hdrs = ["algorithm_checker.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:algorithm_checker"],
)

cc_library(
    name = "alias_passthrough_params",
    hdrs = ["alias_passthrough_params.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:alias_passthrough_params"],
)

cc_library(
    name = "async_wrapper",
    hdrs = ["async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:async_wrapper"],
)

cc_library(
    name = "dynamic_slice_copy",
    hdrs = ["dynamic_slice_copy.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_copy"],
)

cc_library(
    name = "dynamic_slice_copy_fusion_async_wrapper",
    hdrs = ["dynamic_slice_copy_fusion_async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_copy_fusion_async_wrapper"],
)

cc_library(
    name = "gpu_copy_async_wrapper",
    hdrs = ["gpu_copy_async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:gpu_copy_async_wrapper"],
)

cc_library(
    name = "dus_accumulator_zero_init_elimination",
    hdrs = ["dus_accumulator_zero_init_elimination.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dus_accumulator_zero_init_elimination"],
)

cc_library(
    name = "block_scaling_rewriter",
    hdrs = ["block_scaling_rewriter.h"],
    # TODO: Add tags = if_google(["avoid_dep"]) after migrating learning/brain/engprod/xwatch/hlo_selection.
    deps = ["//xla/backends/gpu/hlo/transforms:block_scaling_rewriter"],
)

cc_library(
    name = "conv_padding_legalization",
    hdrs = ["conv_padding_legalization.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:conv_padding_legalization"],
)

cc_library(
    name = "conv_rewriter",
    hdrs = ["conv_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:conv_rewriter"],
)

cc_library(
    name = "conv_fp8_fallback",
    hdrs = ["conv_fp8_fallback.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:conv_fp8_fallback"],
)

cc_library(
    name = "conv_kind_assignment",
    hdrs = ["conv_kind_assignment.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:conv_kind_assignment"],
)

cc_library(
    name = "constant_fill_copy_rewriter",
    hdrs = ["constant_fill_copy_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:constant_fill_copy_rewriter"],
)

cc_library(
    name = "copy_fusion",
    hdrs = ["copy_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:copy_fusion"],
)

cc_library(
    name = "cudnn_fusion_utils",
    hdrs = ["cudnn_fusion_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_fusion_utils"],
)

cc_library(
    name = "cudnn_gemm_dimensions",
    hdrs = ["cudnn_gemm_dimensions.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_gemm_dimensions"],
)

cc_library(
    name = "conv_fusion_rewriter",
    hdrs = ["conv_fusion_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:conv_fusion_rewriter"],
)

cc_library(
    name = "cublas_pad_for_gemms",
    hdrs = ["cublas_pad_for_gemms.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cublas_pad_for_gemms"],
)

cc_library(
    name = "cudnn_custom_call_converter",
    hdrs = ["cudnn_custom_call_converter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_custom_call_converter"],
)

cc_library(
    name = "cudnn_fused_conv_rewriter",
    hdrs = ["cudnn_fused_conv_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_fused_conv_rewriter"],
)

cc_library(
    name = "compiled_cudnn_graph_cache",
    hdrs = ["compiled_cudnn_graph_cache.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:compiled_cudnn_graph_cache"],
)

cc_library(
    name = "cudnn_fusion_compiler",
    hdrs = ["cudnn_fusion_compiler.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_fusion_compiler"],
)

cc_library(
    name = "cudnn_norm_rewriter",
    hdrs = ["cudnn_norm_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_norm_rewriter"],
)

cc_library(
    name = "cudnn_pad_for_convolutions",
    hdrs = ["cudnn_pad_for_convolutions.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_pad_for_convolutions"],
)

cc_library(
    name = "cudnn_simplify_padding",
    hdrs = ["cudnn_simplify_padding.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_simplify_padding"],
)

cc_library(
    name = "cudnn_custom_call_compiler",
    hdrs = ["cudnn_custom_call_compiler.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:cudnn_custom_call_compiler"],
)

cc_library(
    name = "dot_dimension_sorter",
    hdrs = ["dot_dimension_sorter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dot_dimension_sorter"],
)

cc_library(
    name = "dot_normalizer",
    hdrs = ["dot_normalizer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dot_normalizer"],
)

cc_library(
    name = "gpu_transpose_folding",
    hdrs = ["gpu_transpose_folding.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:gpu_transpose_folding"],
)

cc_library(
    name = "dot_operand_converter",
    hdrs = ["dot_operand_converter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dot_operand_converter"],
)

cc_library(
    name = "one_hot_gather_utils",
    hdrs = ["one_hot_gather_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:one_hot_gather_utils"],
)

cc_library(
    name = "dot_strength_reduction",
    hdrs = ["dot_strength_reduction.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dot_strength_reduction"],
)

cc_library(
    name = "double_buffer_loop_unrolling",
    hdrs = ["double_buffer_loop_unrolling.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:double_buffer_loop_unrolling"],
)

cc_library(
    name = "dynamic_slice_fusion",
    hdrs = ["dynamic_slice_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_fusion"],
)

cc_library(
    name = "dynamic_slice_analysis",
    hdrs = ["dynamic_slice_analysis.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_analysis"],
)

cc_library(
    name = "dynamic_slice_annotator",
    hdrs = ["dynamic_slice_annotator.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_annotator"],
)

cc_library(
    name = "dynamic_slice_fusion_rewriter_v2",
    hdrs = ["dynamic_slice_fusion_rewriter_v2.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:dynamic_slice_fusion_rewriter_v2"],
)

cc_library(
    name = "erase_xla_late_frontend_attributes",
    hdrs = ["erase_xla_late_frontend_attributes.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:erase_xla_late_frontend_attributes"],
)

cc_library(
    name = "explicit_collectives_group_async_wrapper",
    hdrs = ["explicit_collectives_group_async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:explicit_collectives_group_async_wrapper"],
)

cc_library(
    name = "group_collectives_by_key",
    hdrs = ["group_collectives_by_key.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:group_collectives_by_key"],
)

cc_library(
    name = "explicit_stream_annotation_async_wrapper",
    hdrs = ["explicit_stream_annotation_async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:explicit_stream_annotation_async_wrapper"],
)

cc_library(
    name = "fusion_wrapper",
    hdrs = ["fusion_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:fusion_wrapper"],
)

cc_library(
    name = "gemm_broadcast_folding_rewriter",
    hdrs = ["gemm_broadcast_folding_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:gemm_broadcast_folding_rewriter"],
)

cc_library(
    name = "gemm_fusion",
    hdrs = ["gemm_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:gemm_fusion"],
)

cc_library(
    name = "gemm_fusion_swap_operands",
    hdrs = ["gemm_fusion_swap_operands.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:gemm_fusion_swap_operands"],
)

cc_library(
    name = "gemm_rewriter",
    hdrs = ["gemm_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:gemm_rewriter"],
)

cc_library(
    name = "sycl_gemm_workspace_pass",
    hdrs = ["sycl_gemm_workspace_pass.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:sycl_gemm_workspace_pass"],
)

cc_library(
    name = "layout_assignment",
    hdrs = ["layout_assignment.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:layout_assignment"],
)

cc_library(
    name = "move_copy_to_users",
    hdrs = ["move_copy_to_users.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:move_copy_to_users"],
)

cc_library(
    name = "multi_output_fusion",
    hdrs = ["multi_output_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:multi_output_fusion"],
)

cc_library(
    name = "multi_phase_fusion",
    hdrs = ["multi_phase_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:multi_phase_fusion"],
)

cc_library(
    name = "bitcast_utils",
    hdrs = ["bitcast_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:bitcast_utils"],
)

cc_library(
    name = "priority_fusion",
    hdrs = ["priority_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:priority_fusion"],
)

cc_library(
    name = "add_tracking_suffix_to_instruction_names",
    hdrs = ["add_tracking_suffix_to_instruction_names.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:add_tracking_suffix_to_instruction_names"],
)

cc_library(
    name = "ragged_dot_fusion_rewriter",
    hdrs = ["ragged_dot_fusion_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:ragged_dot_fusion_rewriter"],
)

cc_library(
    name = "ragged_dot_cudnn_support",
    hdrs = ["ragged_dot_cudnn_support.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:ragged_dot_cudnn_support"],
)

cc_library(
    name = "reduce_scatter_creator",
    hdrs = ["reduce_scatter_creator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:reduce_scatter_creator"],
)

cc_library(
    name = "reduction_degenerate_dim_remover",
    hdrs = ["reduction_degenerate_dim_remover.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:reduction_degenerate_dim_remover"],
)

cc_library(
    name = "reduction_dimension_grouper",
    hdrs = ["reduction_dimension_grouper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:reduction_dimension_grouper"],
)

cc_library(
    name = "reduction_layout_normalizer",
    hdrs = ["reduction_layout_normalizer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:reduction_layout_normalizer"],
)

cc_library(
    name = "reduction_splitter",
    hdrs = ["reduction_splitter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:reduction_splitter"],
)

cc_library(
    name = "rename_fusions",
    hdrs = ["rename_fusions.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:rename_fusions"],
)

cc_library(
    name = "sanitize_constant_names",
    hdrs = ["sanitize_constant_names.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:sanitize_constant_names"],
)

cc_library(
    name = "scalar_constant_sinker",
    hdrs = ["scalar_constant_sinker.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scalar_constant_sinker"],
)

cc_library(
    name = "scaled_dot_rewriter",
    hdrs = ["scaled_dot_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scaled_dot_rewriter"],
)

cc_library(
    name = "composite_rewriter",
    hdrs = ["composite_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:composite_rewriter"],
)

cc_library(
    name = "scatter_determinism_expander",
    hdrs = ["scatter_determinism_expander.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scatter_determinism_expander"],
)

cc_library(
    name = "scatter_slice_simplifier",
    hdrs = ["scatter_slice_simplifier.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scatter_slice_simplifier"],
)

cc_library(
    name = "scatter_expander",
    hdrs = ["scatter_expander.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scatter_expander"],
)

cc_library(
    name = "scheduling_instruction_annotator",
    hdrs = ["scheduling_instruction_annotator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scheduling_instruction_annotator"],
)

cc_library(
    name = "softmax_rewriter_triton",
    hdrs = ["softmax_rewriter_triton.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:softmax_rewriter_triton"],
)

cc_library(
    name = "sort_iota_fusion",
    hdrs = ["sort_iota_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:sort_iota_fusion"],
)

cc_library(
    name = "scan_rewriter",
    hdrs = ["scan_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scan_rewriter"],
)

cc_library(
    name = "sort_rewriter",
    hdrs = ["sort_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:sort_rewriter"],
)

cc_library(
    name = "estimate_cub_sort_scratch_size",
    hdrs = ["estimate_cub_sort_scratch_size.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:estimate_cub_sort_scratch_size"],
)

cc_library(
    name = "deviceless_estimate_cub_sort_scratch_size",
    hdrs = ["deviceless_estimate_cub_sort_scratch_size.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:deviceless_estimate_cub_sort_scratch_size"],
)

cc_library(
    name = "estimate_cub_scan_scratch_size",
    hdrs = ["estimate_cub_scan_scratch_size.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:estimate_cub_scan_scratch_size"],
)

cc_library(
    name = "stream_attribute_annotator",
    hdrs = ["stream_attribute_annotator.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:stream_attribute_annotator"],
)

cc_library(
    name = "stream_attribute_async_wrapper",
    hdrs = ["stream_attribute_async_wrapper.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:stream_attribute_async_wrapper"],
)

cc_library(
    name = "topk_specializer",
    hdrs = ["topk_specializer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:topk_specializer"],
)

cc_library(
    name = "topk_splitter",
    hdrs = ["topk_splitter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:topk_splitter"],
)

cc_library(
    name = "tree_reduction_rewriter",
    hdrs = ["tree_reduction_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:tree_reduction_rewriter"],
)

cc_library(
    name = "triangular_solve_rewriter",
    hdrs = ["triangular_solve_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:triangular_solve_rewriter"],
)

cc_library(
    name = "triton_fusion_numerics_verifier",
    hdrs = ["triton_fusion_numerics_verifier.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:triton_fusion_numerics_verifier"],
)

cc_library(
    name = "variadic_op_splitter",
    hdrs = ["variadic_op_splitter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:variadic_op_splitter"],
)

cc_library(
    name = "windowed_einsum_handler",
    hdrs = ["windowed_einsum_handler.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:windowed_einsum_handler"],
)

cc_library(
    name = "pgle_accuracy_checker",
    hdrs = ["pgle_accuracy_checker.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:pgle_accuracy_checker"],
)

cc_library(
    name = "ragged_all_to_all_decomposer",
    hdrs = ["ragged_all_to_all_decomposer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:ragged_all_to_all_decomposer"],
)

cc_library(
    name = "ragged_all_to_all_multi_host_decomposer",
    hdrs = ["ragged_all_to_all_multi_host_decomposer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:ragged_all_to_all_multi_host_decomposer"],
)

cc_library(
    name = "ragged_all_to_all_canonicalizer",
    hdrs = ["ragged_all_to_all_canonicalizer.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:ragged_all_to_all_canonicalizer"],
)

cc_library(
    name = "splitk_rewriter",
    hdrs = ["splitk_rewriter.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:splitk_rewriter"],
)

cc_library(
    name = "convert_triton_gemm_config",
    hdrs = ["convert_triton_gemm_config.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:convert_triton_gemm_config"],
)

cc_library(
    name = "pdl_launch_annotation",
    hdrs = ["pdl_launch_annotation.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:pdl_launch_annotation"],
)

cc_library(
    name = "scan_rewriter_triton",
    hdrs = ["scan_rewriter_triton.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:scan_rewriter_triton"],
)

cc_library(
    name = "horizontal_fusion",
    hdrs = ["horizontal_fusion.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms/fusion:horizontal_fusion"],
)

cc_library(
    name = "config_assigner_pass",
    hdrs = ["config_assigner_pass.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:config_assigner_pass"],
)

cc_library(
    name = "gpu_cost_model_stats_collection",
    hdrs = ["gpu_cost_model_stats_collection.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:gpu_cost_model_stats_collection"],
)

cc_library(
    name = "sol_gpu_cost_model_stats_collection",
    hdrs = ["sol_gpu_cost_model_stats_collection.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:sol_gpu_cost_model_stats_collection"],
)

cc_library(
    name = "collective_ptable_stats_collection",
    hdrs = ["collective_ptable_stats_collection.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:collective_ptable_stats_collection"],
)

cc_library(
    name = "matmul_ptable_stats_collection",
    hdrs = ["matmul_ptable_stats_collection.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/hlo/transforms:matmul_ptable_stats_collection"],
)
