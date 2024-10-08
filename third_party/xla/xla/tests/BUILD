# Description:
#   Base testing infrastructure for XLA.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla:package_groups.bzl", "xla_tests_package_groups")
load("//xla:xla.bzl", "tests_build_defs_bzl_deps", "xla_cc_binary", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "generate_backend_test_macros", "xla_test", "xla_test_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

xla_tests_package_groups()

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

# Target to add main for tests. Do not link this target and
# //third_party/tensorflow/tsl/platform:test_main into the same target.
cc_library(
    name = "xla_internal_test_main",
    testonly = True,
    srcs = ["xla_internal_test_main.cc"],
    deps = [
        "//xla:debug_options_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
    alwayslink = True,
)

cc_library(
    name = "test_macros_header",
    testonly = True,
    hdrs = ["test_macros.h"],
)

# Generate a test_macros_${BACKEND} library per backend with the proper copts.
generate_backend_test_macros()

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_verifier",
        "//xla/service:transfer_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "literal_test_util",
    testonly = True,
    srcs = ["literal_test_util.cc"],
    hdrs = ["literal_test_util.h"],
    deps = [
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:literal_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/test_utils:verified_hlo_module
# instead.
cc_library(
    name = "verified_hlo_module",
    testonly = True,
    hdrs = ["verified_hlo_module.h"],
    deps = [
        "//xla/hlo/test_utils:verified_hlo_module",
    ],
)

cc_library(
    name = "pjrt_client_registry",
    srcs = ["pjrt_client_registry.cc"],
    hdrs = ["pjrt_client_registry.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_cpu_client_registry",
    testonly = True,
    srcs = [
        "pjrt_cpu_client_registry.cc",
    ],
    deps = [
        ":pjrt_client_registry",
        "//xla/pjrt/cpu:cpu_client",
    ],
)

cc_library(
    name = "pjrt_gpu_client_registry",
    testonly = True,
    srcs = [
        "pjrt_gpu_client_registry.cc",
    ],
    deps = [
        ":pjrt_client_registry",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
    ],
)

cc_library(
    name = "hlo_test_base",
    testonly = True,
    srcs = ["hlo_test_base.cc"],
    hdrs = ["hlo_test_base.h"],
    deps = [
        ":literal_test_util",
        ":pjrt_client_registry",
        ":test_utils",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/test_utils:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:backend",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer_hdr",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service:hlo_verifier",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:filecheck",
        "//xla/tests:verified_hlo_module",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_binary(
    name = "local_client_aot_test_helper",
    srcs = ["local_client_aot_test_helper.cc"],
    deps = [
        "//xla:types",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

genrule(
    name = "local_client_aot_test_computation",
    outs = ["local_client_aot_test_computation.o"],
    cmd = "$(location :local_client_aot_test_helper) $(TARGET_CPU) > $(OUTS)",
    local = 1,
    tools = [":local_client_aot_test_helper"],
)

cc_library(
    name = "client_library_test_base",
    testonly = True,
    srcs = ["client_library_test_base.cc"],
    hdrs = ["client_library_test_base.h"],
    deps = [
        ":literal_test_util",
        ":test_utils",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/tsl/lib/core:bitmap",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "llvm_irgen_test_base",
    testonly = True,
    srcs = ["llvm_irgen_test_base.cc"],
    hdrs = ["llvm_irgen_test_base.h"],
    deps = [
        ":codegen_test_base",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:filecheck",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "codegen_test_base",
    testonly = True,
    srcs = ["codegen_test_base.cc"],
    hdrs = ["codegen_test_base.h"],
    deps = [
        ":hlo_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/test_utils:filecheck
# instead.
cc_library(
    name = "filecheck",
    testonly = True,
    hdrs = ["filecheck.h"],
    deps = [
        "//xla/hlo/test_utils:filecheck",
    ],
)

cc_library(
    name = "local_client_test_base",
    testonly = True,
    srcs = ["local_client_test_base.cc"],
    hdrs = ["local_client_test_base.h"],
    deps = [
        ":client_library_test_base",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:computation_placer",
        "//xla/service:hlo_module_config",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "bad_rng_shape_validation_test",
    srcs = ["bad_rng_shape_validation_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "buffer_donation_test",
    srcs = ["buffer_donation_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:status_macros",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/service:backend",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:verified_hlo_module",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "conv_depthwise_test",
    timeout = "long",
    srcs = [
        "conv_depthwise_test.cc",
    ],
    shard_count = 50,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":conv_depthwise_common",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:despecializer",
        "//xla/service:float_normalization",
    ],
)

xla_test(
    name = "conv_depthwise_backprop_filter_test",
    timeout = "long",
    srcs = ["conv_depthwise_backprop_filter_test.cc"],
    shard_count = 40,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:despecializer",
        "//xla/service:float_normalization",
    ],
)

xla_test(
    name = "grouped_convolution_test",
    timeout = "long",
    srcs = ["grouped_convolution_test.cc"],
    disabled_backends = [
        # disabled because it times out.
        "cpu",
    ],
    shard_count = 50,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:despecializer",
        "//xla/service:float_normalization",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_test(
    name = "check_execution_arity_test",
    srcs = ["check_execution_arity_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "query_inferred_shape_test",
    srcs = ["query_inferred_shape_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "while_test",
    srcs = ["while_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "xla_hlo_profile_test",
    srcs = ["xla_hlo_profile_test.cc"],
    backends = [
        # Hlo profiles are only supported on CPU/GPU.
        "cpu",
        "gpu",
    ],
    tags = ["not_run:arm"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":test_utils",
        "//xla:array2d",
        "//xla:shape_util",
        "//xla:util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:platform_util",
        "//xla/service:stream_pool",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "axpy_simple_test",
    srcs = ["axpy_simple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "map_test",
    srcs = ["map_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/stream_executor",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "params_test",
    timeout = "long",
    srcs = ["params_test.cc"],
    shard_count = 30,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "pred_test",
    srcs = ["pred_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "select_test",
    srcs = ["select_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "conditional_test",
    srcs = ["conditional_test.cc"],
    shard_count = 2,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
    ],
)

xla_test(
    name = "unary_op_test",
    srcs = ["unary_op_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "complex_unary_op_test",
    srcs = [
        "complex_unary_op_samples.h",
        "complex_unary_op_test.cc",
    ],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:math",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "scalar_computations_test",
    srcs = ["scalar_computations_test.cc"],
    shard_count = 32,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "deallocation_test",
    srcs = ["deallocation_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deconstruct_tuple_test",
    srcs = ["deconstruct_tuple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "array_elementwise_ops_test",
    srcs = ["array_elementwise_ops_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    shard_count = 25,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:comparison_util",
        "//xla:fp_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@ml_dtypes//:float8",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "conv_depthwise_common",
    testonly = True,
    srcs = ["conv_depthwise_common.cc"],
    hdrs = ["conv_depthwise_common.h"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:despecializer",
        "//xla/service:float_normalization",
    ],
)

xla_test(
    name = "reduce_precision_test",
    srcs = ["reduce_precision_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "fft_test",
    srcs = ["fft_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

# Repeat fft_test with single-threaded runtime.
xla_test(
    name = "fft_single_threaded_test",
    srcs = ["fft_test.cc"],
    backend_args = {
        "cpu": [
            "--xla_cpu_multi_thread_eigen=false",
        ],
    },
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "dot_operation_test",
    srcs = ["dot_operation_test.cc"],
    shard_count = 20,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:types",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

# Run dot tests with auto-tuning disabled.  This just does a basic sanity check
# that setting xla_gpu_autotune_level to 0 does not break simple graphs.
xla_test(
    name = "dot_operation_test_autotune_disabled",
    srcs = ["dot_operation_test.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 20,
    tags = [
        "optonly",
        # TODO(b/151340488): Timed out on 2020-03-12.
        "nozapfhahn",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

# Run dot tests with dot canonicalization after the layout assignment pass.
xla_test(
    name = "dot_operation_test_canonicalization_after_layout",
    timeout = "long",
    srcs = ["dot_operation_test.cc"],
    args = [
        "--xla_tpu_order_dot_after_layout=true",
    ],
    disabled_backends = [
        "cpu",
        "gpu",
        "interpreter",
    ],
    shard_count = 50,
    tags = [
        "nozapfhahn",
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_test(
    name = "gather_operation_test",
    srcs = ["gather_operation_test.cc"],
    shard_count = 20,
    tags = [
        "test_hlo_pjrt_runner",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:execution_options_util",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/hlo/builder:xla_builder",
        "//xla/service",
    ],
)

xla_test(
    name = "scatter_test",
    srcs = ["scatter_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    # TODO(b/245550554): enable Pjrt runner for scatter test once it's fixed.
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:types",
        "@com_google_absl//absl/strings",
    ],
)

# Repeat dot_operation_runtime_test with single-threaded eigen.
xla_test(
    name = "dot_operation_single_threaded_runtime_test",
    srcs = ["dot_operation_test.cc"],
    backend_args = {
        "cpu": [
            "--xla_cpu_multi_thread_eigen=false",
        ],
    },
    shard_count = 20,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:util",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal_util",
        "//xla:types",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:constants",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
    ],
)

CONVOLUTION_TEST_DEPS = [
    ":test_macros_header",
    "//xla:array2d",
    "//xla:array4d",
    "//xla:literal",
    "//xla:reference_util",
    "//xla:shape_util",
    "@com_google_absl//absl/status:statusor",
    "//xla:util",
    "//xla:xla_data_proto_cc",
    "//xla/client:global_data",
    "//xla/client:local_client",
    "//xla/hlo/builder:padding",
    "//xla/hlo/builder:xla_builder",
    "//xla/tests:client_library_test_base",
    "//xla/tests:hlo_test_base",
    "//xla/tests:literal_test_util",
    "//xla/tests:xla_internal_test_main",
    "@local_tsl//tsl/platform:logging",
    "@local_tsl//tsl/platform:test",
    "//xla/stream_executor/cuda:cuda_platform_id",
]

xla_test(
    name = "convolution_test",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    shard_count = 50,
    tags = [
        "optonly",
        # Timed out on 2020-07-18
        "nozapfhahn",
        "test_xla_cpu_thunks",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_1d",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    # Turn on logging so that VLOG statements don't appear uncovered to zapfhahn.
    args = ["--vmodule=convolution_emitter=7"],
    # In the open source build, convolution_test_1d_gpu fails because it doesn't
    # recognize --vmodule.
    disabled_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 50,
    tags = [
        "cuda-only",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_1d_no_vmodule",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 50,
    tags = [
        "cuda-only",
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

# Run convolution tests with auto-tuning disabled.  This just does a basic
# sanity check that setting xla_gpu_autotune_level to 0 does not break simple
# graphs.
xla_test(
    name = "convolution_test_autotune_disabled",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 40,
    tags = [
        "cuda-only",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_1d_autotune_disabled",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 40,
    tags = [
        "cuda-only",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_gpu_alternative_layout",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    backend_args = {"gpu": ["--xla_backend_extra_options=xla_gpu_experimental_conv_disable_layout_heuristic"]},
    backends = ["gpu"],
    shard_count = 25,
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_1d_gpu_alternative_layout",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    backend_args = {"gpu": ["--xla_backend_extra_options=xla_gpu_experimental_conv_disable_layout_heuristic"]},
    backends = ["gpu"],
    shard_count = 25,
    tags = ["cuda-only"],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_cudnn_frontend_disabled",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    backend_args = {"gpu": ["--xla_gpu_enable_cudnn_frontend=false"]},
    backends = ["gpu"],
    shard_count = 50,
    tags = [
        "nozapfhahn",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_variants_test",
    timeout = "long",
    srcs = ["convolution_variants_test.cc"],
    backend_tags = {
        # TODO(b/31436974): Fix msan failure. Failed on 2016-09-12.
        "cpu": ["nomsan"],
    },
    shard_count = 30,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "convolution_dimension_numbers_test",
    timeout = "long",
    srcs = ["convolution_dimension_numbers_test.cc"],
    shard_count = 20,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:reference_util",
        "//xla:test",
        "//xla/client:local_client",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "convolution_cudnn_test",
    timeout = "long",
    srcs = ["convolution_cudnn_test.cc"],
    backends = [
        "gpu_v100",
        "gpu_a100",
        "gpu_h100",
    ],
    data = ["data/cudnn_reproducer.hlo"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "batch_normalization_test",
    srcs = ["batch_normalization_test.cc"],
    disabled_backends = [
        # BatchNorm HLOs are not handled by the interpreter backend, and the
        # BatchNorm expander is not run on the interpreter.
        "interpreter",
    ],
    shard_count = 40,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/lib/math:math_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "bfloat16_test",
    srcs = ["bfloat16_test.cc"],
    shard_count = 40,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "float8_test",
    srcs = ["float8_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla/hlo/builder:xla_builder",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_test(
    name = "half_test",
    srcs = ["half_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "int4_test",
    srcs = ["int4_test.cc"],
    backends = [
        "cpu",
        "gpu",
        "interpreter",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "slice_test",
    timeout = "long",
    srcs = ["slice_test.cc"],
    shard_count = 40,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:reference_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "multidimensional_slice_test",
    srcs = ["multidimensional_slice_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "dynamic_ops_test",
    timeout = "moderate",
    srcs = ["dynamic_ops_test.cc"],
    shard_count = 4,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:reference_util",
        "//xla:test_helpers",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:computation_placer",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "tuple_test",
    srcs = ["tuple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "vector_ops_reduce_test",
    srcs = ["vector_ops_reduce_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    shard_count = 31,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reduce_window_rewriter_execution_test",
    srcs = ["reduce_window_rewriter_execution_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test",
    ],
)

# External xla_test targets can add "reduce_window_test_library" to xla_test_library_deps, in order
# to refer to the cc_library compiled with the correct backend macros. The following test target
# "reduce_window_test" is an example.
xla_test_library(
    name = "reduce_window_test_library",
    srcs = ["reduce_window_test.cc"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reduce_window_test",
    timeout = "long",
    srcs = [],
    shard_count = 40,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    xla_test_library_deps = [":reduce_window_test_library"],
    deps = [":test_macros_header"],
)

xla_test(
    name = "select_and_scatter_test",
    timeout = "long",
    srcs = ["select_and_scatter_test.cc"],
    shard_count = 25,
    tags = [
        "no_mac",  # b/194731834
        "nozapfhahn",
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:reference_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "copy_test",
    srcs = ["copy_test.cc"],
    tags = [
        "test_hlo_pjrt_runner",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reduce_hlo_test",
    srcs = ["reduce_hlo_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "sort_test",
    srcs = ["sort_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "topk_test",
    srcs = ["topk_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "runtime_topk_test",
    srcs = ["runtime_topk_test.cc"],
    backends = ["cpu"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "token_hlo_test",
    srcs = ["token_hlo_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "call_test",
    srcs = ["call_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    backends = ["cpu"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/ffi",
        "//xla/ffi:execution_context",
        "//xla/ffi:ffi_api",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "binop_scaling_test",
    srcs = ["binop_scaling_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:reference_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "broadcast_simple_test",
    srcs = ["broadcast_simple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:reference_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "fmax_fmin_test",
    srcs = ["fmax_fmin_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/hlo/builder:xla_builder",
        "//xla/service",
    ],
)

xla_test(
    name = "log_test",
    srcs = ["log_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "matrix_ops_simple_test",
    timeout = "long",
    srcs = ["matrix_ops_simple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:test",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_test(
    name = "prng_test",
    timeout = "long",
    srcs = ["prng_test.cc"],
    shard_count = 20,
    # TODO(b/148276347) The test fails on macOS.
    tags = [
        "no_mac",
        "noasan",
        "nosan",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "rng_test",
    srcs = ["rng_test.cc"],
    backends = ["cpu"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:rng_bit_generator_expander",
        "//xla/service:rng_expander",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    shard_count = 30,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "dynamic_reshape_test",
    srcs = ["dynamic_reshape_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],  # TODO(b/345034145): Fix tsan error.
    },
    disabled_backends = ["interpreter"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "reverse_test",
    srcs = ["reverse_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "stochastic_convert_test",
    srcs = ["stochastic_convert_test.cc"],
    backends = ["cpu"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_utils",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "vector_ops_simple_test",
    srcs = ["vector_ops_simple_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/stream_executor",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "convert_test",
    srcs = ["convert_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.cc"],
    disabled_backends = [
        # All reduce is not supported on the interpreter backend.
        "interpreter",
    ],
    tags = [
        "test_hlo_pjrt_runner",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
    ],
)

xla_test(
    name = "collective_ops_test",
    srcs = ["collective_ops_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
        "cpu": [
            "notsan",
        ],
    },
    backends = [
        "gpu",
        "cpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/service:computation_placer",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/tests:verified_hlo_module",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
    ],
)

xla_test(
    name = "collective_pipeline_parallelism_test",
    srcs = ["collective_pipeline_parallelism_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
        "cpu": [
            "notsan",
        ],
    },
    backends = [
        "gpu",
        "cpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "collective_ops_e2e_test",
    srcs = ["collective_ops_e2e_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = [
        "gpu",
    ],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "collective_pipeliner_execution_test",
    srcs = ["collective_pipeliner_execution_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:collective_pipeliner",
        "//xla/service:hlo_dce",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "replicated_io_feed_test",
    srcs = ["replicated_io_feed_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "bitcast_convert_test",
    srcs = ["bitcast_convert_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/stream_executor",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "matmul_test",
    srcs = ["matmul_test.cc"],
    backends = [
        "gpu",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
    ],
)

xla_test(
    name = "floor_ceil_test",
    srcs = ["floor_ceil_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "multithreaded_compilation_test",
    srcs = ["multithreaded_compilation_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "value_inference_test",
    srcs = ["value_inference_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:global_data",
        "//xla/hlo/builder:value_inference",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:prng",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "compute_constant_test",
    srcs = ["compute_constant_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:global_data",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "replay_test",
    srcs = ["replay_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "broadcast_test",
    srcs = ["broadcast_test.cc"],
    tags = [
        "test_hlo_pjrt_runner",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "llvm_compiler_test",
    srcs = ["llvm_compiler_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        "//xla:literal_util",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:backend",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "round_trip_packed_literal_test",
    srcs = ["round_trip_packed_literal_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:packed_literal_reader",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "cpu_gpu_fusion_test",
    srcs = ["cpu_gpu_fusion_test.cc"],
    backends = [
        "cpu",
        "gpu",
        "interpreter",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "multioutput_fusion_test",
    srcs = ["multioutput_fusion_test.cc"],
    backends = ["gpu"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "local_client_aot_test",
    srcs = [
        "local_client_aot_test.cc",
        ":local_client_aot_test_computation.o",
    ],
    linkstatic = 1,
    tags = [
        "not_run:arm",  # b/341355246
        "test_xla_cpu_thunks",
    ],
    deps = [
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "local_client_allocation_test",
    srcs = ["local_client_allocation_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:local_service",
        "//xla/service:shaped_buffer",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "local_client_execute_test",
    # TODO(b/79375911): Test times out in LLVM at normal size.
    size = "large",
    srcs = ["local_client_execute_test.cc"],
    shard_count = 30,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:sharding_builder",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/host:host_platform_id",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "outfeed_in_nested_computation_test",
    srcs = ["outfeed_in_nested_computation_test.cc"],
    disabled_backends = [
        # Outfeed ops are not supported on the interpreter backend.
        "interpreter",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":local_client_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_cc_test(
    name = "hlo_metadata_test",
    srcs = [
        "hlo_metadata_test.cc",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":local_client_test_base",
        "//xla:test_helpers",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:cpu_plugin",
        "//xla/service:local_service",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "round_trip_transfer_test",
    srcs = ["round_trip_transfer_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:literal",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reshape_motion_test",
    srcs = ["reshape_motion_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "deep_graph_test",
    srcs = ["deep_graph_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
    ],
)

xla_cc_test(
    name = "literal_test_util_test",
    srcs = ["literal_test_util_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":literal_test_util",
        "//xla:literal",
        "//xla:test_helpers",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "transfer_manager_test",
    srcs = ["transfer_manager_test.cc"],
    shard_count = 50,
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:generic_transfer_manager",
        "//xla/service:shaped_buffer",
        "//xla/service:stream_pool",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

# A demo of textual IR based test.
xla_test(
    name = "sample_text_test",
    srcs = ["sample_text_test.cc"],
    # You can leave this empty if you want to test all supported backends.
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla:types",
    ],
)

# A demo of test that loads an hlo module from a file and compares results on gpu and cpu.
xla_test(
    name = "sample_file_test",
    srcs = ["sample_file_test.cc"],
    backends = ["gpu"],
    data = ["isolated_convolution.hlo"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:test",
        "//xla/service:cpu_plugin",  # reference backend
        "//xla/service:platform_util",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    # There is nothing backend specific in this test, so just pick an arbitrary backend.
    backends = ["cpu"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":local_client_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_test(
    name = "iota_test",
    srcs = ["iota_test.cc"],
    shard_count = 50,
    tags = [
        # Require optimized builds, iota_test_cpu is very slow in fastbuild.
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_cc_test(
    name = "multiple_devices_on_host_test",
    srcs = ["multiple_devices_on_host_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:cpu_plugin",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "ptxas_bug_120501638",
    srcs = ["ptxas_bug_120501638.cc"],
    tags = [
        # Disabled in OSS until nvidia publicly releases a fixed ptxas.
        "no_oss",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:debug_options_flags",
        "//xla:test",
    ],
)

xla_test(
    name = "get_dimension_size_test",
    srcs = ["get_dimension_size_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "set_dimension_size_test",
    srcs = ["set_dimension_size_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],  # TODO(b/345034145): Fix tsan error.
    },
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "triangular_solve_test",
    srcs = ["triangular_solve_test.cc"],
    real_hardware_only = True,
    shard_count = 3,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:array2d",
        "//xla:literal",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "cholesky_test",
    srcs = ["cholesky_test.cc"],
    shard_count = 10,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
    ],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "constant_reduction_function_test",
    srcs = ["constant_reduction_function_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:test",
        "//xla:types",
    ],
)

xla_cc_test(
    name = "tile_assignment_test",
    srcs = ["tile_assignment_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:test",
        "//xla/hlo/ir:tile_assignment",
        "@com_google_absl//absl/hash",
    ],
)

xla_test(
    name = "numerics_test",
    srcs = ["numerics_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//xla:literal_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "concatenate_test",
    srcs = ["concatenate_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],
    },
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "batch_norm_grad_test",
    srcs = ["batch_norm_grad_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "batch_norm_training_test",
    srcs = ["batch_norm_training_test.cc"],
    tags = ["test_xla_cpu_thunks"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

bzl_library(
    name = "plugin_bzl",
    srcs = ["plugin.bzl"],
    deps = [],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        ":plugin_bzl",
        "//xla:xla_bzl",
        "//xla/stream_executor:build_defs_bzl",
        "@local_tsl//tsl/platform:build_config_root_bzl",
    ] + tests_build_defs_bzl_deps(),
)
