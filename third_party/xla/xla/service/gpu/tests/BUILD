# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

# copybara:uncomment load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite_for_gpus")
load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

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
    tags = tf_gpu_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:executable",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "hlo_pjrt_gpu_test_base",
    testonly = True,
    srcs = ["hlo_pjrt_gpu_test_base.cc"],
    hdrs = ["hlo_pjrt_gpu_test_base.h"],
    deps = [
        "//xla/backends/gpu/target_config",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:aot_utils",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tests:pjrt_client_registry",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_test(
    name = "dynamic_slice_fusion_test",
    srcs = if_gpu_is_configured(["dynamic_slice_fusion_test.cc"]),
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = [
        "no-oneapi",  # TODO(intel-tf): Enable this test for SYCL when IntelGpuCompiler is implemented.
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = if_gpu_is_configured(
        #keep sorted
        [
            "//xla:error_spec",
            "//xla:shape_util",
            "//xla/backends/gpu:ffi",
            "//xla/ffi",
            "//xla/ffi:ffi_api",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/status",
            "@tsl//tsl/platform:test",
        ],
    ) + [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "async_kernel_launch_test",
    srcs = ["async_kernel_launch_test.cc"],
    backends = ["gpu"],
    # "requires-net:external" tag allows uploading `xprof` results.
    tags = if_google(["requires-net:external"]) + ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "command_buffer_test",
    srcs = ["command_buffer_test.cc"],
    backends = ["gpu"],
    # TSAN reports a false positive for the test due to noninstrumented CUDA library code.
    env = {"TSAN_OPTIONS": "ignore_noninstrumented_modules=1"},
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "async_command_buffer_test",
    srcs = ["async_command_buffer_test.cc"],
    tags = tf_gpu_tests_tags() + ["pjrt_migration_candidate"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "float_conversions_test",
    srcs = ["float_conversions_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    #TODO(b/450135639): Remove timeout override once the test is fixed.
    timeout = "moderate",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "swap_conv_operands_test",
    srcs = [
        "swap_conv_operands_test.cc",
    ],
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:device_description",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "parallel_reduction_test",
    srcs = [
        "parallel_reduction_test.cc",
    ],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_compilation_parallelism_test",
    srcs = [
        "gpu_compilation_parallelism_test.cc",
    ],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_copy_alone_test",
    srcs = [
        "gpu_copy_alone_test.cc",
    ],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_triton_custom_call_test",
    srcs = ["gpu_triton_custom_call_test.cc"],
    backends = [
        "a100",
        "h100",
        "v100",
        "b200",
        "amdgpu_any",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",  # build_cleaner: keep
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:client_library_test_base",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
    ],
)

xla_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    backends = [
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:hlo_module_config",
        "//xla/service:platform_util",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_alignment_test",
    srcs = ["gpu_alignment_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_googletest//:gtest_main",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "ragged_dot_test",
    srcs = ["ragged_dot_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "regression_dot_test",
    srcs = ["regression_dot_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    shard_count = 15,
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu/transforms:sort_rewriter",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

lit_test_suite_for_gpus(
    name = "hlo_lit_tests",
    srcs = enforce_glob(
        [
            "bitcast-convert.hlo",
            "calling_convention.hlo",
            "dot_bf16.hlo",
            "kernel_reuse.hlo",
            "offload_scan_output.hlo",
            "pad_to_static.hlo",
            "reduce_fold_zero_add.hlo",
            "reduce-precision.hlo",
            "rng_get_and_update_state.hlo",
            "single_instruction.hlo",
            "slice_to_dynamic.hlo",
            "sorting.hlo",
            "sub_byte_collectives.hlo",
            "triton_calling_convention.hlo",
            "triton_naming.hlo",
            "zero_clamp_abs_index.hlo",
        ],
        include = [
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    default_tags = ["gpu"],  # Needs to run in a build with a gpu configured.
    disabled_on_gpus = {
        "v100": [
            "kernel_reuse.hlo",
            "triton_calling_convention.hlo",
            "triton_naming.hlo",
        ],
        "p100": [
            "kernel_reuse.hlo",
            "triton_calling_convention.hlo",
            "triton_naming.hlo",
        ],
        "mi200": [
            "element_wise_row_vectorization.hlo",
            "scatter_bf16.hlo",
            "single_instruction.hlo",
            "reduce_unnested.hlo",
            "reduction_vectorization_sm_all.hlo",
        ],
    },
    gpus = [
        "a100_pcie_80",
        "a6000",
        "b200",
        "h100_sxm",
        "mi200",
        "p100",
        "v100",
    ],
    hermetic_cuda_data_dir = "%S/../../../../../cuda_nvcc",
    tags = ["no-oneapi"],
    tools = [
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

# copybara:uncomment_begin(triton-opt tool doesn't build in OSS)
# cc_binary(
#     name = "xla-opt",
#     srcs = ["xla-opt.cc"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:AllExtensions",
#         "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:FuncExtensions",
#         "@llvm-project//mlir:LLVMIRTransforms",
#         "@llvm-project//mlir:LLVMToLLVMIRTranslation",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:MlirOptLib",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:RegisterAllExtensions",  # buildcleaner: keep
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@stablehlo//:stablehlo_ops",
#         "//xla/backends/gpu/codegen/emitters/transforms:passes",
#         "//xla/backends/gpu/codegen/triton:compilation_pipeline",
#         "//xla/backends/gpu/codegen/triton/ir:triton_xla",
#         "//xla/backends/gpu/codegen/triton/transforms:passes",
#         "//xla/codegen/emitters/ir:xla",
#         "//xla/codegen/emitters/transforms:passes",
#         "//xla/codegen/xtile/ir:xtile",
#         "//xla/stream_executor:device_description",
#         "//xla/stream_executor/cuda:cuda_compute_capability",
#         "@triton//:AllPassesAndDialects",
#         "@triton//:TritonNvidiaGPUTransforms",
#         "@triton//third_party/amd:TestAMDAnalysis",  # buildcleaner: keep
#     ],
# )
# copybara:uncomment_end

xla_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "mock_custom_call_test",
    srcs = ["mock_custom_call_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:xla_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "dynamic_shared_memory_test",
    srcs = if_cuda_is_configured(["dynamic_shared_memory_test.cc"]),
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ] + if_cuda_is_configured([
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_address",
    ]),
)

xla_test(
    name = "tensor_float_32_global_var_test",
    srcs = ["tensor_float_32_global_var_test.cc"],
    backends = [
        "a100",
        "b200",
        "amdgpu_any",
    ] + if_oss([
        "nvgpu_any",
    ]),
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:tensor_float_32_utils",
    ],
)

xla_test(
    name = "gpu_cub_sort_test",
    size = "medium",
    srcs = ["gpu_cub_sort_test.cc"],
    backends = ["gpu"],
    shard_count = 15,
    tags = [
        "nodebug",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/service/gpu/transforms:sort_rewriter",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_fused_mha_test",
    srcs = ["gpu_fused_mha_test.cc"],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
        # TODO(b/445172709): Re-enable once fixed.
        "h100": ["broken"],
    },
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    shard_count = 2,
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# This library is here to be reused by tests.
cc_library(
    name = "simple_optimization_test",
    testonly = True,
    srcs = ["simple_optimization_test.cc"],
    tags = tf_gpu_tests_tags(),
    deps = [
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,  # This library registers test cases at static initialization time.
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":simple_optimization_test",
        "//xla/tests:xla_internal_test_main",
    ],
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":simple_optimization_test",
        "//xla/tests:xla_internal_test_main",
    ],
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":simple_optimization_test",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "gpu_int4_test",
    srcs = ["gpu_int4_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplify_fp_conversions_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "nop_custom_call_test",
    srcs = ["nop_custom_call_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "ptx_kernel_test",
    srcs = ["ptx_kernel_test.cc"],
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        "//xla:literal",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_ops_e2e_test_base",
    testonly = True,
    srcs = ["collective_ops_e2e_test_base.cc"],
    hdrs = ["collective_ops_e2e_test_base.h"],
    deps = [
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/target_config",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
        "@com_google_protobuf//:protobuf",
    ] + if_google([
        # In OSS this dependency is added automatically for xla_test targets. See
        # third_party/tensorflow/compiler/xla/xla.default.bzl.
        "//xla/tsl/framework:allocator",
    ]),
)

xla_test(
    name = "collective_ops_command_buffer_test",
    srcs = ["collective_ops_command_buffer_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
