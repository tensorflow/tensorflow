# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.

load("//xla/tests:build_defs.bzl", "xla_test")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:glob_lit_test.bzl", "glob_lit_tests")
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_codegen_test",
    testonly = True,
    srcs = ["gpu_codegen_test.cc"],
    hdrs = ["gpu_codegen_test.h"],
    tags = tf_cuda_tests_tags(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:types",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor",
        "//xla/tests:filecheck",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
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
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_reduce_scatter_creator_test",
    srcs = ["gpu_reduce_scatter_creator_test.cc"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:gpu_reduce_scatter_creator",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
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
        "@com_google_absl//absl/strings",
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
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_module_config",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gemm_rewrite_test",
    srcs = if_cuda_is_configured(["gemm_rewrite_test.cc"]) + if_rocm_is_configured(["gemm_rewrite_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:filecheck",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
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
        "//xla:error_spec",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gemm_broadcast_folding_rewriter",
        "//xla/service/gpu:gemm_rewriter",
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
        "//xla:statusor",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/lib/core:status_test_util",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_degenerate_dim_remover",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_layout_normalizer",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:tree_reduction_rewriter",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:logging",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
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
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_dimension_grouper",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
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
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
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
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
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
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
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
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:instruction_fusion",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_pipeline_test",
    srcs = ["gpu_fusion_pipeline_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service/gpu:fusion_merger",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:instruction_fusion",
        "//xla/service/gpu:multi_output_fusion",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
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
        "//xla/tests:llvm_irgen_test_base",
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
        "//xla/hlo/ir:hlo",
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
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
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
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
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
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:compile_module_to_llvm_ir",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service/gpu:target_constants",
        "//xla/service/gpu/llvm_gpu_backend",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor/host:host_platform",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_module_loader",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/util:command_line_flags",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/service/gpu:nvptx_compiler_impl",
        "//xla/stream_executor/cuda:cublas_plugin",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/stream_executor/rocm:rocblas_plugin",
        "//xla/stream_executor/rocm:rocm_helpers",
        "//xla/service/gpu:amdgpu_compiler_impl",
        "@local_tsl//tsl/platform:rocm_rocdl_path",
    ]),
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags() + [
    ],
    driver = "//xla:run_lit.sh",
    exclude = ["execute_memzero_thunk.mlir"],
    features = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=SUBST_TIDX=@llvm.nvvm.read.ptx.sreg.tid.x",
        "--param=SUBST_CTAIDX=@llvm.nvvm.read.ptx.sreg.ctaid.x",
        "--param=SUBST_BARRIER=@llvm.nvvm.barrier0",
        "--param=SUBST_KERNEL_ANNOTATION=''",
        "--param=SUBST_ADDRSPACE_ANNOTATION=''",
    ]) + if_rocm_is_configured([
        "--param=PTX=GCN",
        "--param=SUBST_TIDX=@llvm.amdgcn.workitem.id.x",
        "--param=SUBST_CTAIDX=@llvm.amdgcn.workgroup.id.x",
        "--param=SUBST_BARRIER=@llvm.amdgcn.s.barrier",
        "--param=SUBST_KERNEL_ANNOTATION=amdgpu_kernel[SPACE]",
        "--param=SUBST_ADDRSPACE_ANNOTATION='addrspace(5)[SPACE]'",
    ]),
    tags_override = {
        "reduction_vectorization_sm_all.hlo": ["no_rocm"],
        "element_wise_row_vectorization.hlo": ["no_rocm"],
        "single_instruction.hlo": ["no_rocm"],
        "reduce_unnested.hlo": ["no_rocm"],
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
        "@llvm-project//mlir:run_lit.sh",
    ],
    visibility = ["//visibility:public"],
)

xla_cc_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
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
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
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
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/gpu:asm_compiler",
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
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fused_mha_test",
    srcs = ["gpu_fused_mha_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:array4d",
        "//xla:literal",
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
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

# This library is here to be reused by tests.
cc_library(
    name = "simple_optimization_test",
    testonly = True,
    srcs = ["simple_optimization_test.cc"],
    tags = tf_cuda_tests_tags(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
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
