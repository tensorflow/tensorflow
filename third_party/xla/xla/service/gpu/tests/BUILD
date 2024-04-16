# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_gpu_tests_tags",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "gpu_codegen_test",
    testonly = True,
    srcs = ["gpu_codegen_test.cc"],
    hdrs = ["gpu_codegen_test.h"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla/service:executable",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:filecheck",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "float_conversions_test",
    srcs = ["float_conversions_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_reduce_scatter_creator_test",
    srcs = ["gpu_reduce_scatter_creator_test.cc"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:gpu_reduce_scatter_creator",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "gpu_all_gather_optimizer_test",
    srcs = ["gpu_all_gather_optimizer_test.cc"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_all_gather_optimizer",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gemm_rewrite_test",
    srcs = ["gemm_rewrite_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:statusor",
        "//xla:test",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:variant_visitor",
        "//xla/stream_executor:device_description",
        "//xla/tests:filecheck",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_cc_test(
    name = "gemm_broadcast_folding_rewrite_test",
    srcs = ["gemm_broadcast_folding_rewrite_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = tf_gpu_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gemm_broadcast_folding_rewriter",
        "//xla/service/gpu:gemm_rewriter",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_degenerate_dim_remover_test",
    srcs = [
        "reduction_degenerate_dim_remover_test.cc",
    ],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_degenerate_dim_remover",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_layout_normalizer_test",
    srcs = [
        "reduction_layout_normalizer_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_layout_normalizer",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = [
        "tree_reduction_rewriter_test.cc",
    ],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:tree_reduction_rewriter",
        "//xla/stream_executor:device_description",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "swap_conv_operands_test",
    srcs = [
        "swap_conv_operands_test.cc",
    ],
    tags = ["no_rocm"] + tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:gpu_plugin",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:device_description",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_dimension_grouper_test",
    srcs = [
        "reduction_dimension_grouper_test.cc",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_dimension_grouper",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "parallel_reduction_test",
    srcs = [
        "parallel_reduction_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_parser",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_compilation_parallelism_test",
    srcs = [
        "gpu_compilation_parallelism_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_alone_test",
    srcs = [
        "gpu_copy_alone_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_triton_custom_call_test",
    srcs = ["gpu_triton_custom_call_test.cc"],
    backends = [
        "gpu_a100",
        "gpu_v100",
    ],
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",  # build_cleaner: keep
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-nvidia",
        "requires-gpu-sm60-only",
    ]},
    backends = [
        "gpu",
        "gpu_p100",
    ],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "concatenate_emitter_test",
    srcs = ["concatenate_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "transpose_emitter_test",
    srcs = ["transpose_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_emitter_test",
    srcs = ["reduction_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:instruction_fusion",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_pipeline_test",
    srcs = ["gpu_fusion_pipeline_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service/gpu:fusion_merger",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:instruction_fusion",
        "//xla/service/gpu:multi_output_fusion",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_alignment_test",
    testonly = True,
    srcs = ["gpu_alignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:custom_call_target_registry",
        "//xla/service:gpu_plugin",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/tests:filecheck",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:filecheck",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_input_fusible_slice_test",
    srcs = ["gpu_input_fusible_slice_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_convolution_regression_test",
    srcs = ["gpu_convolution_regression_test.cc"],
    backend_args = {"gpu": [
        "--xla_enable_hlo_passes_only=layout-assignment,gpu-conv-algorithm-picker",
        "--xla_gpu_crash_on_verification_failures",
    ]},
    backends = ["gpu"],
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "add_preds.hlo",
            "calling_convention.hlo",
            "copy.hlo",
            "dynamic_update_slice_inplace.hlo",
            "element_wise_row_vectorization.hlo",
            "fused_scatter.hlo",
            "fused_slice.hlo",
            "kernel_reuse.hlo",
            "launch_dimensions.hlo",
            "pad_to_static.hlo",
            "reduce_atomic_min.hlo",
            "reduce_column_layout_change.hlo",
            "reduce_f64_column.hlo",
            "reduce_large_row_to_scalar.hlo",
            "reduce_row_vectorized.hlo",
            "reduce_unnested.hlo",
            "reduce_variadic_column.hlo",
            "reduction_vectorization_sm_all.hlo",
            "rng_get_and_update_state.hlo",
            "scatter.hlo",
            "select_and_scatter.hlo",
            "single_instruction.hlo",
            "slice_to_dynamic.hlo",
            "sorting.hlo",
            "transpose_021.hlo",
            "transpose_021_extra_output.hlo",
            "transpose_210.hlo",
            "transpose_210_extra_output.hlo",
            "triton_naming.hlo",
        ],
        include = [
            "*.hlo",
        ],
    ),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a6000",
    ]) + if_rocm_is_configured([
        "--param=PTX=GCN",
        "--param=GPU=mi200",
    ]),
    cfg = "//xla:lit.cfg.py",
    data = [
        ":test_utilities",
    ],
    default_tags = tf_cuda_tests_tags(),
    tags_override = {
        "reduction_vectorization_sm_all.hlo": ["no_rocm"],
        "element_wise_row_vectorization.hlo": ["no_rocm"],
        "single_instruction.hlo": ["no_rocm"],
        "reduce_unnested.hlo": ["no_rocm"],
    },
    tools = [
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//xla/tools:hlo-opt",
        "//xla/tools/hlo_opt:gpu_specs/a100_80.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/a6000.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/h100.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/mi200.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/p100.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/v100.txtpb",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mock_custom_call_test",
    srcs = ["mock_custom_call_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "dynamic_shared_memory_test",
    srcs = if_cuda_is_configured(["dynamic_shared_memory_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/stream_executor:platform_manager",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_asm_compiler",
        "//xla/service/gpu:gpu_asm_opts_util",
        "//xla/stream_executor",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_memory",
    ]),
)

xla_test(
    name = "tensor_float_32_global_var_test",
    srcs = ["tensor_float_32_global_var_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-nvidia",
        "requires-gpu-sm80-only",
    ]},
    backends = [
        "gpu",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_fused_mha_test",
    srcs = ["gpu_fused_mha_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-sm80",
    ]},
    backends = [
        "gpu",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    shard_count = 2,
    deps = [
        ":gpu_codegen_test",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

# This library is here to be reused by tests.
cc_library(
    name = "simple_optimization_test",
    testonly = True,
    srcs = ["simple_optimization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

# This shows that tests can load an autotune cache.
#
# If the GPU used for running the test is different from the one in the cache, then the cache will
# be loaded, but not used.
xla_test(
    name = "load_autotune_results_using_execpath_test",
    srcs = [],
    backends = ["gpu"],
    # Data dependency must be declared for the cache.
    data = ["test_autotune_cache.textproto"],
    env = {"XLA_FLAGS": "--xla_gpu_load_autotune_results_from=" +
                        "$(execpath test_autotune_cache.textproto)"},
    deps = [":simple_optimization_test"],
)

# This shows that tests can load an autotune cache using the TEST_WORKSPACE prefix.
#
# This also works from the command line, by specifying this arguments:
# --test_env=XLA_FLAGS=--xla_gpu_load_autotune_results_from=TEST_WORKSPACE/my/package/autotune_results_test.textproto.
#
# If the GPU used for running the test is different from the one in the cache, then the cache will
# be loaded, but not used.
xla_test(
    name = "load_autotune_results_from_test_workspace_test",
    srcs = [],
    backends = ["gpu"],
    # Data dependency must be declared for the cache.
    data = ["test_autotune_cache.textproto"],
    env = {"XLA_FLAGS": "--xla_gpu_load_autotune_results_from=TEST_WORKSPACE/" +
                        package_name() +
                        "/test_autotune_cache.textproto"},
    deps = [":simple_optimization_test"],
)

# This shows that tests can dump an autotune cache into their output directory.
#
# This also works from the command line, by specifying these arguments:
# --test_env=XLA_FLAGS=--xla_gpu_dump_autotune_results_to=TEST_UNDECLARED_OUTPUTS_DIR/autotune_cache.textproto
# --test_sharding_strategy=disabled
xla_test(
    name = "dump_autotune_results_to_test_outputs_test",
    srcs = [],
    backends = ["gpu"],
    env = {"XLA_FLAGS": "--xla_gpu_dump_autotune_results_to=" +
                        "TEST_UNDECLARED_OUTPUTS_DIR/autotune_cache.textproto"},
    # Sharding must be disabled to correctly dump the autotune cache for all test.
    shard_count = 1,
    deps = [":simple_optimization_test"],
)

xla_cc_test(
    name = "gpu_int4_test",
    srcs = ["gpu_int4_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplify_fp_conversions_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:xla_proto_cc",
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)
