# Description:
#   Base testing infrastructure for XLA.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla:package_groups.bzl", "xla_test_friend_package_group")
load("//xla:xla.default.bzl", "tests_build_defs_bzl_deps", "xla_cc_test", "xla_internal")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "xla_test", "xla_test_library")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

xla_test_friend_package_group(name = "friends")

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

# Target to add main for tests. Do not link this target and
# @com_google_googletest//:gtest_main into the same target.
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
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
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
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
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
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
    alwayslink = True,
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
    alwayslink = True,
)

cc_library(
    name = "pjrt_interpreter_client_registry",
    testonly = True,
    srcs = [
        "pjrt_interpreter_client_registry.cc",
    ],
    deps = [
        ":pjrt_client_registry",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/interpreter:interpreter_client",
    ],
    alwayslink = True,
)

cc_library(
    name = "hlo_test_base",
    testonly = True,
    srcs = ["hlo_test_base.cc"],
    hdrs = ["hlo_test_base.h"],
    deps = [
        ":hlo_runner_agnostic_reference_mixin",
        ":hlo_runner_agnostic_test_base",
        ":pjrt_client_registry",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/pjrt:pjrt_client",
        "//xla/service:backend",
        "//xla/service:computation_placer_hdr",
        "//xla/service:executable",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "hlo_runner_agnostic_test_base",
    testonly = True,
    srcs = ["hlo_runner_agnostic_test_base.cc"],
    hdrs = ["hlo_runner_agnostic_test_base.h"],
    deps = [
        ":literal_test_util",
        ":test_utils",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "hlo_runner_agnostic_reference_mixin",
    testonly = True,
    srcs = ["hlo_runner_agnostic_reference_mixin.cc"],
    hdrs = ["hlo_runner_agnostic_reference_mixin.h"],
    deps = [
        ":hlo_runner_agnostic_test_base",
        ":literal_test_util",
        ":test_utils",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_pjrt_interpreter_reference_mixin",
    testonly = True,
    hdrs = ["hlo_pjrt_interpreter_reference_mixin.h"],
    deps = [
        ":hlo_runner_agnostic_reference_mixin",
        "//xla/pjrt/interpreter:interpreter_client",
        "//xla/service:hlo_runner_pjrt",
    ],
)

cc_library(
    name = "hlo_pjrt_test_base",
    testonly = True,
    srcs = ["hlo_pjrt_test_base.cc"],
    hdrs = ["hlo_pjrt_test_base.h"],
    deps = [
        ":hlo_runner_agnostic_test_base",
        ":hlo_runner_pjrt_test_utils",
        ":pjrt_client_registry",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/service:hlo_runner_interface",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "client_library_test_base",
    testonly = True,
    srcs = ["client_library_test_base.cc"],
    hdrs = ["client_library_test_base.h"],
    deps = [
        ":client_library_test_runner_utils",
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
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:bitmap",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,  # This library registers test cases at static initialization time.
)

cc_library(
    name = "client_library_test_runner_mixin",
    testonly = True,
    hdrs = ["client_library_test_runner_mixin.h"],
    deps = [
        ":client_library_test_runner_utils",
        ":hlo_runner_agnostic_test_base",
        ":literal_test_util",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "client_library_test_runner_utils",
    testonly = True,
    srcs = ["client_library_test_runner_utils.cc"],
    hdrs = ["client_library_test_runner_utils.h"],
    deps = [
        ":test_utils",
        "//xla:array2d",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llvm_irgen_test_base",
    testonly = True,
    srcs = ["llvm_irgen_test_base.cc"],
    hdrs = ["llvm_irgen_test_base.h"],
    deps = [
        ":codegen_test_base",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
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

cc_library(
    name = "local_client_test_base",
    testonly = True,
    srcs = ["local_client_test_base.cc"],
    hdrs = ["local_client_test_base.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer",
        "//xla/service:hlo_module_config",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:stream_pool",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

xla_cc_test(
    name = "bad_rng_shape_validation_test",
    srcs = ["bad_rng_shape_validation_test.cc"],
    deps = [
        ":xla_internal_test_main",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "buffer_donation_test",
    srcs = ["buffer_donation_test.cc"],
    backend_args = if_google(
        {
            "cpu": [
                # TODO(b/372312816): Fix the leak in the test.
                "--heap_check=",
            ],
            "interpreter": [
                # TODO(b/372312816): Fix the leak in the test.
                "--heap_check=",
            ],
        },
        {},
    ),
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:status_macros",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:backend",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "conv_depthwise_test",
    timeout = "long",
    srcs = [
        "conv_depthwise_test.cc",
    ],
    shard_count = 30,
    deps = [
        ":client_library_test_base",
        ":conv_depthwise_common",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:float_normalization",
    ],
)

xla_test(
    name = "conv_depthwise_backprop_filter_test",
    timeout = "long",
    srcs = ["conv_depthwise_backprop_filter_test.cc"],
    shard_count = 40,
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:float_normalization",
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
    shard_count = 23,
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:float_normalization",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_test(
    name = "check_execution_arity_test",
    srcs = ["check_execution_arity_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "query_inferred_shape_test",
    srcs = ["query_inferred_shape_test.cc"],
    deps = [
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "while_test",
    srcs = ["while_test.cc"],
    # placeholder for extra args for while_test
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
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
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "axpy_simple_test",
    srcs = ["axpy_simple_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "map_test",
    srcs = ["map_test.cc"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "params_test",
    timeout = "long",
    srcs = ["params_test.cc"],
    shard_count = 15,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "pred_test",
    srcs = ["pred_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "select_test",
    srcs = ["select_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/service",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "conditional_test",
    srcs = ["conditional_test.cc"],
    shard_count = 2,
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "unary_op_test",
    srcs = ["unary_op_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:math",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "scalar_computations_test",
    srcs = ["scalar_computations_test.cc"],
    shard_count = 32,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deallocation_test",
    srcs = ["deallocation_test.cc"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deconstruct_tuple_test",
    srcs = ["deconstruct_tuple_test.cc"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "array_elementwise_ops_test",
    srcs = ["array_elementwise_ops_test.cc"],
    shard_count = 25,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:fp_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_depthwise_common",
    testonly = True,
    srcs = ["conv_depthwise_common.cc"],
    hdrs = ["conv_depthwise_common.h"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_for_library",
    ],
)

xla_test(
    name = "reduce_precision_test",
    srcs = ["reduce_precision_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_test(
    name = "fft_test",
    srcs = ["fft_test.cc"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
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
    ],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "dot_operation_test",
    timeout = "long",
    srcs = ["dot_operation_test.cc"],
    shard_count = 20,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
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
        "//xla:types",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
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
    # placeholder for extra args for dot_operation_test_canonicalization_after_layout
    disabled_backends = [
        "cpu",
        "gpu",
        "interpreter",
    ],
    shard_count = 50,
    tags = [
        "nozapfhahn",
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:error_spec",
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/service",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "get_default_platform_test",
    srcs = ["get_default_platform_test.cc"],
    disabled_backends = [
        "cpu",
        "gpu",
        "interpreter",
    ],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        "//xla/service:platform_util",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "scatter_test",
    srcs = ["scatter_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/service:hlo_module_config",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

# Repeat dot_operation_runtime_test with single-threaded eigen.
xla_test(
    name = "dot_operation_single_threaded_runtime_test",
    timeout = "long",
    srcs = ["dot_operation_test.cc"],
    backend_args = {
        "cpu": [
            "--xla_cpu_multi_thread_eigen=false",
        ],
    },
    shard_count = 20,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:util",
        "//xla/hlo/builder:xla_builder",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:constants",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_test(
    name = "convolution_test",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    shard_count = 50,
    tags = [
        "optonly",
        # Timed out on 2020-07-18
        "nozapfhahn",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
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
    shard_count = 40,
    tags = [
        "cuda-only",
        "optonly",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
    ],
)

xla_test(
    name = "convolution_test_gpu_alternative_layout",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    backend_args = {"gpu": ["--xla_backend_extra_options=xla_gpu_experimental_conv_disable_layout_heuristic"]},
    backends = ["gpu"],
    shard_count = 25,
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

xla_test(
    name = "convolution_test_1d_gpu_alternative_layout",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    backend_args = {"gpu": ["--xla_backend_extra_options=xla_gpu_experimental_conv_disable_layout_heuristic"]},
    backends = ["gpu"],
    shard_count = 25,
    tags = [
        "cuda-only",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "convolution_dimension_numbers_test",
    timeout = "long",
    srcs = ["convolution_dimension_numbers_test.cc"],
    shard_count = 4,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "convolution_cudnn_test",
    timeout = "long",
    srcs = ["convolution_cudnn_test.cc"],
    backends = [
        "v100",
        "a100",
        "h100",
        "b200",
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
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
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
    shard_count = 5,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "float8_test",
    srcs = ["float8_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
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
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "slice_test",
    timeout = "long",
    srcs = ["slice_test.cc"],
    shard_count = 40,
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "multidimensional_slice_test",
    srcs = ["multidimensional_slice_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "dynamic_ops_test",
    timeout = "moderate",
    srcs = ["dynamic_ops_test.cc"],
    shard_count = 4,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ] + if_oss(["not_run:arm"]),
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/service:computation_placer",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

xla_test(
    name = "tuple_test",
    srcs = ["tuple_test.cc"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "vector_ops_reduce_test",
    srcs = ["vector_ops_reduce_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    shard_count = 31,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
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
    deps = [
        ":hlo_test_base",
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
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,  # This library registers test cases at static initialization time.
)

xla_test(
    name = "reduce_window_test",
    timeout = "long",
    srcs = [],
    shard_count = 50,
    tags = [
        "optonly",
    ],
    xla_test_library_deps = [":reduce_window_test_library"],
    deps = [
        ":xla_internal_test_main",
    ],
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
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
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "copy_test",
    srcs = ["copy_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "reduce_hlo_test",
    srcs = ["reduce_hlo_test.cc"],
    backends = [
        "gpu",
        "cpu",
        "interpreter",
    ],
    deps = [
        ":hlo_test_base",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "sort_test",
    srcs = ["sort_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "topk_test",
    srcs = ["topk_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "runtime_topk_test",
    srcs = ["runtime_topk_test.cc"],
    backends = ["cpu"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "token_hlo_test",
    srcs = ["token_hlo_test.cc"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "call_test",
    srcs = ["call_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    backends = ["cpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
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
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

xla_test(
    name = "binop_scaling_test",
    srcs = ["binop_scaling_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:reference_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "broadcast_simple_test",
    srcs = ["broadcast_simple_test.cc"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "fmax_fmin_test",
    srcs = ["fmax_fmin_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla/hlo/builder:xla_builder",
        "//xla/service",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "log_test",
    srcs = ["log_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array3d",
        "//xla:error_spec",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "matrix_ops_simple_test",
    timeout = "long",
    srcs = ["matrix_ops_simple_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
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
    ],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:rng_bit_generator_expander",
        "//xla/hlo/transforms/expanders:rng_expander",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    shard_count = 30,
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_test(
    name = "dynamic_reshape_test",
    srcs = ["dynamic_reshape_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],  # TODO(b/345034145): Fix tsan error.
    },
    disabled_backends = ["interpreter"],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "reverse_test",
    srcs = ["reverse_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":client_library_test_runner_utils",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "stochastic_convert_test",
    srcs = ["stochastic_convert_test.cc"],
    backends = ["cpu"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "vector_ops_simple_test",
    srcs = ["vector_ops_simple_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":client_library_test_runner_utils",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "convert_test",
    srcs = ["convert_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform:ml_dtypes",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
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
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer",
        "//xla/service:hlo_module_config",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:blocking_counter",
        "@ml_dtypes_py//ml_dtypes:float8",
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
    },
    backends = [
        "gpu",
    ],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "nccl_group_execution_test",
    srcs = ["nccl_group_execution_test.cc"],
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
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        ":hlo_runner_agnostic_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "collective_pipeliner_execution_test",
    srcs = ["collective_pipeliner_execution_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:collective_pipeliner",
        "//xla/service:collective_pipeliner_utils",
        "//xla/service:hlo_verifier",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "bitcast_convert_test",
    srcs = ["bitcast_convert_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:ml_dtypes",
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
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
    ],
)

xla_test(
    name = "floor_ceil_test",
    srcs = ["floor_ceil_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "multithreaded_compilation_test",
    srcs = ["multithreaded_compilation_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "compute_constant_test",
    srcs = ["compute_constant_test.cc"],
    deps = [
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "replay_test",
    srcs = ["replay_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "broadcast_test",
    srcs = ["broadcast_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal_util",
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
    deps = [
        ":hlo_test_base",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:backend",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "round_trip_packed_literal_test",
    srcs = ["round_trip_packed_literal_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:packed_literal_reader",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
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
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:protobuf",
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

xla_test(
    name = "local_client_allocation_test",
    srcs = ["local_client_allocation_test.cc"],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
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
    ],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:sharding_builder",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
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
    deps = [
        ":local_client_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
    ],
)

xla_cc_test(
    name = "hlo_metadata_test",
    srcs = [
        "hlo_metadata_test.cc",
    ],
    deps = [
        ":local_client_test_base",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:cpu_plugin",
        "//xla/service:local_service",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "round_trip_transfer_test",
    srcs = ["round_trip_transfer_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:literal",
        "//xla/client:local_client",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reshape_motion_test",
    srcs = ["reshape_motion_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "deep_graph_test",
    srcs = ["deep_graph_test.cc"],
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
    ],
)

xla_cc_test(
    name = "literal_test_util_test",
    srcs = ["literal_test_util_test.cc"],
    deps = [
        ":literal_test_util",
        "//xla:literal",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "transfer_manager_test",
    srcs = ["transfer_manager_test.cc"],
    shard_count = 15,
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:generic_transfer_manager",
        "//xla/service:shaped_buffer",
        "//xla/service:stream_pool",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests:xla_test_backend_predicates",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
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
        "//xla/hlo/testlib:test",
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
    deps = [
        ":local_client_test_base",
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
    timeout = "long",
    srcs = ["iota_test.cc"],
    backend_tags = {
        "cpu": ["optonly"],
    },
    shard_count = 50,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_cc_test(
    name = "multiple_devices_on_host_test",
    srcs = ["multiple_devices_on_host_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "get_dimension_size_test",
    srcs = ["get_dimension_size_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
    ],
)

xla_test(
    name = "set_dimension_size_test",
    srcs = ["set_dimension_size_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],  # TODO(b/345034145): Fix tsan error.
    },
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
    ],
)

xla_test(
    name = "triangular_solve_test",
    srcs = ["triangular_solve_test.cc"],
    real_hardware_only = True,
    shard_count = 3,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "cholesky_test",
    srcs = ["cholesky_test.cc"],
    shard_count = 10,
    tags = [
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/testlib:test",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "constant_reduction_function_test",
    srcs = ["constant_reduction_function_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/testlib:test",
    ],
)

xla_cc_test(
    name = "tile_assignment_test",
    srcs = ["tile_assignment_test.cc"],
    deps = [
        ":xla_internal_test_main",
        "//xla:array3d",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/hash",
    ],
)

xla_test(
    name = "numerics_test",
    srcs = ["numerics_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "concatenate_test",
    srcs = ["concatenate_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],
    },
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "batch_norm_grad_test",
    srcs = ["batch_norm_grad_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
    ],
)

xla_test(
    name = "batch_norm_training_test",
    srcs = ["batch_norm_training_test.cc"],
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
    ],
)

bzl_library(
    name = "plugin_bzl",
    srcs = ["plugin.bzl"],
    deps = ["//xla/tsl:package_groups_bzl"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        ":plugin_bzl",
        "//xla:xla_bzl",
        "//xla/stream_executor:build_defs_bzl",
        "//xla/tsl:package_groups_bzl",
        "//xla/tsl/platform:build_config_root_bzl",
        "//xla/tsl/platform/default:build_config_bzl",
    ] + tests_build_defs_bzl_deps(),
)

xla_test(
    name = "atan2_test",
    srcs = ["atan2_test.cc"],
    deps = [
        ":client_library_test_base",
        "//xla:error_spec",
        "//xla/hlo/builder:xla_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "remainder_test",
    srcs = ["remainder_test.cc"],
    deps = [
        ":client_library_test_base",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla/hlo/builder:xla_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "two_plus_two_simple_test",
    srcs = ["two_plus_two_simple_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/service",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "hlo_runner_pjrt_test_utils",
    testonly = True,
    srcs = ["hlo_runner_pjrt_test_utils.cc"],
    hdrs = ["hlo_runner_pjrt_test_utils.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xla_test_backend_predicates",
    testonly = True,
    srcs = ["xla_test_backend_predicates.cc"],
    hdrs = ["xla_test_backend_predicates.h"],
    visibility = if_google(
        xla_internal(["tests:__pkg__"]),
        ["//visibility:public"],
    ),
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
