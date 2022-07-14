# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.
#
# TODO(jlebar): None of these tests actually use the GPU, so they should not
# need to run on machines with GPUs present.

load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

package(
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
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
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
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core/common_runtime/gpu:gpu_init",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

tf_cc_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "fusion_logical_index_test",
    srcs = ["fusion_logical_index_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "mlir_fft_test",
    srcs = ["mlir_fft_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "mlir_gemm_test",
    srcs = ["mlir_gemm_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "mlir_sorting_test",
    srcs = ["mlir_sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "mlir_gpu_compile_test",
    srcs = ["mlir_gpu_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_reduce_scatter_creator_test",
    srcs = ["gpu_reduce_scatter_creator_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service/gpu:gpu_reduce_scatter_creator",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gemm_rewrite_test",
    srcs = [
        "gemm_rewrite_test.cc",
    ],
    tags = tf_cuda_tests_tags() + [
        "no_rocm",
    ],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "reduction_degenerate_dim_remover_test",
    srcs = [
        "reduction_degenerate_dim_remover_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_degenerate_dim_remover",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "reduction_layout_normalizer_test",
    srcs = [
        "reduction_layout_normalizer_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_layout_normalizer",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = [
        "tree_reduction_rewriter_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:tree_reduction_rewriter",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "swap_conv_operands_test",
    srcs = [
        "swap_conv_operands_test.cc",
    ],
    tags = ["no_rocm"] + tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gemm_rewriter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "reduction_dimension_grouper_test",
    srcs = [
        "reduction_dimension_grouper_test.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:reduction_dimension_grouper",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "parallel_reduction_test",
    srcs = [
        "parallel_reduction_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_copy_alone_test",
    srcs = [
        "gpu_copy_alone_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
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
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_fusible",
        "//tensorflow/compiler/xla/service/gpu:instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = tf_cuda_tests_tags() + ["no_rocm"],
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gpu_input_fusible_slice_test",
    srcs = ["gpu_input_fusible_slice_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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

xla_test(
    name = "bef_executable_test",
    srcs = ["bef_executable_test.cc"],
    backends = ["gpu"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_binary(
    name = "hlo_to_llvm_ir",
    srcs = ["hlo_to_llvm_ir.cc"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        "@llvm-project//llvm:Target",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_device_info",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler_impl",
        "//tensorflow/compiler/xla/service/gpu:nvptx_helper",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/service/gpu:amdgpu_compiler_impl",
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
        "reduction_vectorization_sm_all.hlo": ["no_rocm"],
        "element_wise_row_vectorization.hlo": ["no_rocm"],
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

tf_cc_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

py_test(
    name = "mnist",
    srcs = ["mnist.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "no_oss",
        "notap",
        "requires-gpu-nvidia",
        "requires-net:external",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)
