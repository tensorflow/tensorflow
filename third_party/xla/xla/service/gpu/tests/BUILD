# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
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
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss")
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

xla_test(
    name = "dynamic_slice_fusion_test",
    srcs = if_gpu_is_configured(["dynamic_slice_fusion_test.cc"]),
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = ["notsan"],  # TODO(b/345034145): Fix tsan error.
    deps = if_gpu_is_configured(
        #keep sorted
        [
            "//xla:error_spec",
            "//xla:shape_util",
            "//xla/ffi",
            "//xla/ffi:ffi_api",
            "//xla/stream_executor",
            "//xla/tests:hlo_test_base",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/status",
            "@local_tsl//tsl/platform:test",
        ],
    ) + [
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "async_kernel_launch_test",
    srcs = ["async_kernel_launch_test.cc"],
    # "requires-net:external" tag allows uploading `xprof` results.
    tags = if_google(["requires-net:external"]) + tf_cuda_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "async_command_buffer_test",
    srcs = ["async_command_buffer_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "swap_conv_operands_test",
    srcs = [
        "swap_conv_operands_test.cc",
    ],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "parallel_reduction_test",
    srcs = [
        "parallel_reduction_test.cc",
    ],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_compilation_parallelism_test",
    srcs = [
        "gpu_compilation_parallelism_test.cc",
    ],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    backends = ["gpu"],
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

xla_test(
    name = "gpu_copy_alone_test",
    srcs = [
        "gpu_copy_alone_test.cc",
    ],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    backends = ["gpu"],
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
        "gpu_amd_any",
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

xla_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",  # build_cleaner: keep
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/tests:client_library_test_base",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    backends = [
        "gpu_p100",
        "gpu_amd_any",
    ] + if_oss(["gpu_any"]),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "concatenate_emitter_test",
    srcs = ["concatenate_emitter_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "transpose_emitter_test",
    srcs = ["transpose_emitter_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "reduction_emitter_test",
    srcs = ["reduction_emitter_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu/transforms:instruction_fusion",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_fusion_pipeline_test",
    srcs = ["gpu_fusion_pipeline_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/transforms:fusion_merger",
        "//xla/service/gpu/transforms:instruction_fusion",
        "//xla/service/gpu/transforms:multi_output_fusion",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_alignment_test",
    srcs = ["gpu_alignment_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_input_fusible_slice_test",
    srcs = ["gpu_input_fusible_slice_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:hlo_module_config",
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

xla_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
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
    name = "hlo_lit_tests",
    srcs = enforce_glob(
        [
            "add_preds.hlo",
            "calling_convention.hlo",
            "dot_bf16.hlo",
            "dynamic_update_slice_inplace.hlo",
            "fused_scatter.hlo",
            "fused_slice.hlo",
            "kernel_reuse.hlo",
            "pad_to_static.hlo",
            "reduce_atomic_min.hlo",
            "reduce_column_layout_change.hlo",
            "reduce_f64_column.hlo",
            "reduce_large_row_to_scalar.hlo",
            "reduce_row_vectorized.hlo",
            "reduce_to_scalar_vectorized.hlo",
            "reduce_unnested.hlo",
            "reduce_variadic_column.hlo",
            "reduction_vectorization_sm_all.hlo",
            "rng_get_and_update_state.hlo",
            "scatter.hlo",
            "scatter_bf16.hlo",
            "select_and_scatter.hlo",
            "single_instruction.hlo",
            "slice_to_dynamic.hlo",
            "sorting.hlo",
            "transpose_021.hlo",
            "transpose_021_extra_output.hlo",
            "transpose_10.hlo",
            "transpose_210.hlo",
            "transpose_210_extra_output.hlo",
            "triton_naming.hlo",
            "zero_clamp_abs_index.hlo",
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
        "//xla/tools/hlo_opt:gpu_specs/a100_pcie_80.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/a6000.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/h100_sxm.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/mi200.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/p100.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/v100.txtpb",
    ],
    default_tags = tf_cuda_tests_tags(),
    hermetic_cuda_data_dir = "%S/../../../../../cuda_nvcc",
    tags_override = {
        "element_wise_row_vectorization.hlo": ["cuda-only"],
        "scatter_bf16.hlo": ["cuda-only"],
        "single_instruction.hlo": ["cuda-only"],
        "reduce_unnested.hlo": ["cuda-only"],
        "reduction_vectorization_sm_all.hlo": ["cuda-only"],
    },
    tools = [
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

# copybara:uncomment_begin(triton-opt tool doesn't build in OSS)
# lit_test_suite(
#     name = "mlir_lit_tests",
#     srcs = glob(["*.mlir"]),
#     cfg = "//xla:lit.cfg.py",
#     tools = [
#         ":xla-opt",
#         "@llvm-project//llvm:FileCheck",
#         "@triton//:triton-opt",
#     ],
# )
#
# cc_binary(
#     name = "xla-opt",
#     srcs = ["xla-opt.cc"],
#     deps = [
#         "//xla/service/gpu/fusions/transforms:passes",
#         "//xla/service/gpu/fusions/triton:passes",
#         "@llvm-project//mlir:AllExtensions",
#         "@llvm-project//mlir:MlirOptLib",
#         "@triton//:AllPassesAndDialects",
#     ],
# )
# copybara:uncomment_end

xla_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "mock_custom_call_test",
    srcs = ["mock_custom_call_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla:debug_options_flags",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "dynamic_shared_memory_test",
    srcs = if_cuda_is_configured(["dynamic_shared_memory_test.cc"]),
    backends = ["gpu"],
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
    backends = [
        "gpu_a100",
        "gpu_amd_any",
    ] + if_oss(["gpu_any"]),
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_sparse_dot_test",
    srcs = if_cuda_is_configured(["gpu_sparse_dot_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
    ],
    tags = ["cuda-only"],
    deps = if_cuda_is_configured(
        [
            ":gpu_codegen_test",
            "@com_google_googletest//:gtest",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/types:span",
            "//xla:literal",
            "//xla:literal_util",
            "//xla/tests:xla_internal_test_main",
            "//xla/tsl/lib/core:status_test_util",
        ],
        ["@local_tsl//tsl/platform:test_main"],  # b/317293391
    ),
)

xla_test(
    name = "gpu_cub_sort_test",
    size = "medium",
    srcs = ["gpu_cub_sort_test.cc"],
    backends = ["gpu"],
    shard_count = 15,
    tags = ["nodebug"],
    deps = [
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu/transforms:sort_rewriter",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_fused_mha_test",
    srcs = ["gpu_fused_mha_test.cc"],
    backends = [
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 2,
    deps = [
        ":gpu_codegen_test",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
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
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
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

xla_test(
    name = "gpu_int4_test",
    srcs = ["gpu_int4_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_codegen_test",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplify_fp_conversions_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "nop_custom_call_test",
    srcs = ["nop_custom_call_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)
