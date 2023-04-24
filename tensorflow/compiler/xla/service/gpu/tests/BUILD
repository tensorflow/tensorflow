# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.

load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load(
    "//tensorflow/compiler/xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "mlir_gpu_test_base",
    testonly = True,
    srcs = ["mlir_gpu_test_base.cc"],
    hdrs = ["mlir_gpu_test_base.h"],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_init",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

xla_cc_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_fft_test",
    srcs = ["mlir_fft_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_test(
    name = "mlir_gemm_test",
    srcs = ["mlir_gemm_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-nvidia",
        "requires-gpu-sm80-only",
    ]},
    backends = [
        "gpu",
    ],
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_sorting_test",
    srcs = ["mlir_sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_gpu_compile_test",
    srcs = ["mlir_gpu_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_reduce_scatter_creator_test",
    srcs = ["gpu_reduce_scatter_creator_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service/gpu:gpu_reduce_scatter_creator",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
    ],
)

xla_cc_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gemm_rewrite_test",
    srcs = if_cuda_is_configured(["gemm_rewrite_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gemm_broadcast_folding_rewrite_test",
    srcs = [
        "gemm_broadcast_folding_rewrite_test.cc",
    ],
    tags = tf_cuda_tests_tags() + [
        "no_rocm",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu:gemm_broadcast_folding_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_degenerate_dim_remover_test",
    srcs = [
        "reduction_degenerate_dim_remover_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_degenerate_dim_remover",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "reduction_layout_normalizer_test",
    srcs = [
        "reduction_layout_normalizer_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_layout_normalizer",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = [
        "tree_reduction_rewriter_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:tree_reduction_rewriter",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "reduction_dimension_grouper_test",
    srcs = [
        "reduction_dimension_grouper_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_dimension_grouper",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "transpose_emitter_test",
    srcs = ["transpose_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "reduction_emitter_test",
    srcs = ["reduction_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_device_info_for_tests",
        "//tensorflow/compiler/xla/service/gpu:gpu_fusible",
        "//tensorflow/compiler/xla/service/gpu:instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_pipeline_test",
    srcs = ["gpu_fusion_pipeline_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service/gpu:fusion_merger",
        "//tensorflow/compiler/xla/service/gpu:gpu_device_info_for_tests",
        "//tensorflow/compiler/xla/service/gpu:instruction_fusion",
        "//tensorflow/compiler/xla/service/gpu:multi_output_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_alignment_test",
    testonly = True,
    srcs = ["gpu_alignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_input_fusible_slice_test",
    srcs = ["gpu_input_fusible_slice_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

build_test(
    name = "hlo_to_llvm_ir_build_test",
    targets = [
        ":hlo_to_llvm_ir",
    ],
)

xla_cc_binary(
    name = "hlo_to_llvm_ir",
    testonly = True,
    srcs = ["hlo_to_llvm_ir.cc"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service/gpu:compile_module_to_llvm_ir",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_device_info_for_tests",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor:device_description_proto_cc_impl",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/util:command_line_flags",
        "@llvm-project//llvm:Target",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler_impl",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
    ]) + if_rocm_is_configured([
        "//tensorflow/tsl/platform:rocm_rocdl_path",
        "//tensorflow/compiler/xla/service/gpu:amdgpu_compiler_impl",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocblas_plugin",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_helpers",
    ]),
)

glob_lit_tests(
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags() + [
        "no_pip",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = ["execute_memzero_thunk.mlir"],
    tags_override = {
        "add_preds.hlo": ["no_rocm"],
        "concat.hlo": ["no_rocm"],
        "calling_convention.hlo": ["no_rocm"],
        "copy.hlo": ["no_rocm"],
        "dynamic_update_slice_inplace.hlo": ["no_rocm"],
        "element_wise_row_vectorization.hlo": ["no_rocm"],
        "fused_scatter.hlo": ["no_rocm"],
        "fused_slice.hlo": ["no_rocm"],
        "fused_slice_different_operands.hlo": ["no_rocm"],
        "fusion.hlo": ["no_rocm"],
        "kernel_reuse.hlo": ["no_rocm"],
        "launch_dimensions.hlo": ["no_rocm"],
        "pad_to_static.hlo": ["no_rocm"],
        "reduce_unnested.hlo": ["no_rocm"],
        "reduction_vectorization_sm_all.hlo": ["no_rocm"],
        "rng_get_and_update_state.hlo": ["no_rocm"],
        "scatter.hlo": ["no_rocm"],
        "select_and_scatter.hlo": ["no_rocm"],
        "single_instruction.hlo": ["no_rocm"],
        "slice_to_dynamic.hlo": ["no_rocm"],
	"triton_naming.hlo": ["no_rocm"],
        "copy_nested.hlo": ["no_rocm"],
        "sorting.hlo": ["no_rocm"],
        "add_preds_amdgcn.hlo": ["no_cuda_asan"],
        "concat_amdgcn.hlo": ["no_cuda_asan"],
        "calling_convention_amdgcn.hlo": ["no_cuda_asan"],
        "copy_amdgcn.hlo": ["no_cuda_asan"],
        "copy_nested_amdgcn.hlo": ["no_cuda_asan"],
        "dynamic_update_slice_inplace_amdgcn.hlo": ["no_cuda_asan"],
        "fused_scatter_amdgcn.hlo": ["no_cuda_asan"],
        "fused_slice_amdgcn.hlo": ["no_cuda_asan"],
        "fused_slice_different_operands_amdgcn.hlo": ["no_cuda_asan"],
        "fusion_amdgcn.hlo": ["no_cuda_asan"],
        "launch_dimensions_amdgcn.hlo": ["no_cuda_asan"],
        "pad_to_static_amdgcn.hlo": ["no_cuda_asan"],
        "reduce_unnested_amdgcn.hlo": ["no_cuda_asan"],
        "rng_get_and_update_state_amdgcn.hlo": ["no_cuda_asan"],
        "scatter_amdgcn.hlo": ["no_cuda_asan"],
        "select_and_scatter_amdgcn.hlo": ["no_cuda_asan"],
        "single_instruction_amdgcn.hlo": ["no_cuda_asan"],
        "slice_to_dynamic_amdgcn.hlo": ["no_cuda_asan"],
	"triton_naming_amdgcn.hlo": ["no_cuda_asan"],
        "sorting_amdgcn.hlo": ["no_cuda_asan"],
    },
    test_file_exts = ["hlo"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":hlo_to_llvm_ir",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "dynamic_shared_memory_test",
    srcs = if_cuda_is_configured(["dynamic_shared_memory_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
    ]),
)
