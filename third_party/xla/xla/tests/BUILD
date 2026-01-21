# Description:
#   Base testing infrastructure for XLA.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla:package_groups.bzl", "xla_test_friend_package_group")
load("//xla:xla.default.bzl", "tests_build_defs_bzl_deps", "xla_cc_test", "xla_internal")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

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
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
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
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_verifier",
        "//xla/service:transfer_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:protobuf",
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
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
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
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/pjrt:pjrt_client",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
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
        ":aot_utils",
        ":hlo_runner_agnostic_reference_mixin",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/service:hlo_runner_pjrt",
    ],
)

cc_library(
    name = "hlo_pjrt_test_base",
    testonly = True,
    srcs = ["hlo_pjrt_test_base.cc"],
    hdrs = ["hlo_pjrt_test_base.h"],
    deps = [
        ":aot_utils",
        ":hlo_runner_agnostic_test_base",
        ":pjrt_client_registry",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
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
        "//xla:array",
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
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
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
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status_macros",
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
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "codegen_test_base",
    testonly = True,
    srcs = ["codegen_test_base.cc"],
    hdrs = ["codegen_test_base.h"],
    deps = [
        ":hlo_test_base_with_mlir_context",
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
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:comparison_util",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:backend",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service:maybe_owning_device_address",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "conv_depthwise_test",
    timeout = "long",
    srcs = [
        "conv_depthwise_test.cc",
    ],
    shard_count = 30,
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":conv_depthwise_common",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:float_normalization",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
    ],
)

xla_test(
    name = "conv_depthwise_backprop_filter_test",
    timeout = "long",
    srcs = ["conv_depthwise_backprop_filter_test.cc"],
    shard_count = 40,
    tags = [
        "nozapfhahn",
        "pjrt_migration_candidate",
    ],  # TODO: b/445492433 - Re-enable when it doesn't time out.
    use_legacy_runtime = True,
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
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
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deallocation_test",
    srcs = ["deallocation_test.cc"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":client_library_test_base",
        ":xla_internal_test_main",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deconstruct_tuple_test",
    srcs = ["deconstruct_tuple_test.cc"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
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
    deps = [
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "dot_operation_test",
    timeout = "long",
    srcs = ["dot_operation_test.cc"],
    aot_compile_test = False,
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    shard_count = 20,
    tags = [
        "optonly",
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
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

# Run dot tests with auto-tuning disabled.  This just does a basic sanity check
# that setting xla_gpu_autotune_level to 0 does not break simple graphs.
xla_test(
    name = "dot_operation_autotune_disabled_test",
    srcs = ["dot_operation_test.cc"],
    aot_compile_test = False,
    args = ["--xla_gpu_autotune_level=0"],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    backends = ["gpu"],
    shard_count = 20,
    tags = [
        "optonly",
        # TODO(b/151340488): Timed out on 2020-03-12.
        "nozapfhahn",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ] + if_rocm_is_configured([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

# Run dot tests with dot canonicalization after the layout assignment pass.
xla_test(
    name = "dot_operation_canonicalization_after_layout_test",
    timeout = "long",
    srcs = ["dot_operation_test.cc"],
    aot_compile_test = False,
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
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
        "//xla/tests:xla_test_backend_predicates",
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
        ":hlo_pjrt_test_base",
        "//xla/service:platform_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
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

xla_test(
    name = "scatter_deterministic_expander_test",
    srcs = ["scatter_test.cc"],
    args = [
        "--xla_gpu_deterministic_ops",
        "--xla_gpu_enable_scatter_determinism_expander",
    ],
    backends = ["gpu"],
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
    aot_compile_test = False,
    backend_args = {
        "cpu": [
            "--xla_cpu_multi_thread_eigen=false",
        ],
    },
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    shard_count = 50,
    tags = [
        "optonly",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
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
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:ml_dtypes",
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
    name = "convolution_1d_test",
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
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
    ],
)

xla_test(
    name = "convolution_1d_no_vmodule_test",
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
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
    ],
)

# Run convolution tests with auto-tuning disabled.  This just does a basic
# sanity check that setting xla_gpu_autotune_level to 0 does not break simple
# graphs.
xla_test(
    name = "convolution_autotune_disabled_test",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 40,
    tags = [
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
    name = "convolution_1d_autotune_disabled_test",
    timeout = "long",
    srcs = ["convolution_test_1d.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 40,
    tags = [
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
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:test",
        "@eigen_archive//:eigen3",
    ],
)

xla_test(
    name = "convolution_gpu_alternative_layout_test",
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
    name = "convolution_1d_gpu_alternative_layout_test",
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
        "//xla/tests:xla_test_backend_predicates",
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
        "//xla/tests:xla_test_backend_predicates",
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "@tsl//tsl/platform:ml_dtypes",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(b/475838852): Re-enable once fixed.
xla_test(
    name = "slice_test",
    timeout = "long",
    srcs = ["slice_test.cc"],
    shard_count = 50,
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
    aot_compile_test = False,
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
        "//xla/tests:xla_test_backend_predicates",
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/service",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "reduce_window_rewriter_execution_test",
    srcs = ["reduce_window_rewriter_execution_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "reduce_window_test_library",
    testonly = True,
    srcs = ["reduce_window_test.cc"],
    # This is set intentionally as to avoid the default behavior of the TSL
    # `cc_library` definition that is used in this file.
    compatible_with = [],
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
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
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
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":reduce_window_test_library",
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
        "//xla/tests:xla_test_backend_predicates",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner_interface",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
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
        "//xla/backends/cpu:ffi",
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

# TODO(b/475838852): Re-enable once fixed.
xla_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:array",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
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
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/service",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
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
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

xla_test(
    name = "dynamic_reshape_test",
    srcs = ["dynamic_reshape_test.cc"],
    backend_tags = {
        "gpu": ["notsan"],  # TODO(b/345034145): Fix tsan error.
    },
    disabled_backends = ["interpreter"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
        "@tsl//tsl/platform:statusor",
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
        "@tsl//tsl/platform:ml_dtypes",
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
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/status:status_matchers",
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "@ml_dtypes_py//ml_dtypes:float8",
        "@tsl//tsl/platform:blocking_counter",
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "@tsl//tsl/platform:statusor",
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":hlo_pjrt_test_base",
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
        ],
    },
    backends = [
        "gpu",
    ],
    deps = [
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:array",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:pattern_matcher",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/tests:collective_ops_e2e_test_base",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
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

cc_library(
    name = "collective_ops_ffi_kernels",
    hdrs = ["collective_ops_ffi_kernels.h"],
    deps = [
        "//xla:types",
        "//xla/stream_executor:kernel",
    ],
)

cuda_library(
    name = "collective_ops_ffi_kernels_cuda",
    srcs = ["collective_ops_ffi_kernels.cu.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":collective_ops_ffi_kernels",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/tsl/cuda:nccl",
        "@com_google_absl//absl/base",
    ],
    alwayslink = True,  # static kernel registation
)

xla_cc_test(
    name = "collective_ops_ffi_kernels_test",
    srcs = ["collective_ops_ffi_kernels_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":collective_ops_ffi_kernels",
        "//xla:future",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/backends/gpu/collectives:gpu_collectives_plugin",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:symmetric_memory",
        "//xla/runtime:device_id",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ] + if_cuda_is_configured([
        ":collective_ops_ffi_kernels_cuda",
    ]),
)

xla_test(
    name = "collective_ops_ffi_test",
    srcs = ["collective_ops_ffi_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
        ],
    },
    backends = ["gpu"],
    deps = [
        ":collective_ops_ffi_kernels",
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:future",
        "//xla:literal",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu:ffi",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_execution",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:reduction_kind",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu/tests:collective_ops_e2e_test_base",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + if_cuda_is_configured([
        ":collective_ops_ffi_kernels_cuda",
    ]),
)

xla_test(
    name = "collective_metadata_test",
    srcs = ["collective_metadata_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
        ],
    },
    backends = [
        "gpu",
    ],
    deps = [
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service/gpu/tests:collective_ops_e2e_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "collective_ops_sharded_unsharded_e2e_test",
    srcs = ["collective_ops_sharded_unsharded_e2e_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
        ],
    },
    backends = [
        "gpu",
    ],
    deps = [
        ":literal_test_util",
        ":test_utils",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/tests:collective_ops_e2e_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:regexp",
    ],
)

xla_test(
    name = "ragged_all_to_all_e2e_test",
    srcs = ["ragged_all_to_all_e2e_test.cc"],
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
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:backend",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:pattern_matcher",
        "//xla/service:platform_util",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_memory_space_assignment",
        "//xla/service/gpu/tests:collective_ops_e2e_test_base",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/integrations:device_mem_allocator",
        "//xla/stream_executor/integrations:stream_executor_allocator",
        "//xla/stream_executor/integrations:tf_allocator_adapter",
        "//xla/tsl/framework:bfc_allocator",
        "//xla/tsl/framework:device_id",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:regexp",
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
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

xla_test(
    name = "matmul_test",
    srcs = ["matmul_test.cc"],
    backends = [
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "compute_constant_test",
    srcs = ["compute_constant_test.cc"],
    use_legacy_runtime = True,
    deps = [
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/testlib:test",
        "//xla/stream_executor:platform",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
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
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
    ],
)

xla_test(
    name = "llvm_compiler_test",
    srcs = ["llvm_compiler_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:backend",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "round_trip_packed_literal_test",
    srcs = ["round_trip_packed_literal_test.cc"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:packed_literal_reader",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:local_client",
        "//xla/service",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
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
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_test(
    name = "multioutput_fusion_test",
    srcs = ["multioutput_fusion_test.cc"],
    backends = ["gpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":literal_test_util",
        ":pjrt_client_registry",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_test(
    name = "local_client_allocation_test",
    srcs = ["local_client_allocation_test.cc"],
    use_legacy_runtime = True,
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
        "@tsl//tsl/platform:test",
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
    use_legacy_runtime = True,
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
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/tests:xla_test_backend_predicates",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

xla_test(
    name = "outfeed_in_nested_computation_test",
    srcs = ["outfeed_in_nested_computation_test.cc"],
    disabled_backends = [
        # Outfeed ops are not supported on the interpreter backend.
        "interpreter",
    ],
    use_legacy_runtime = True,
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
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//xla:array4d",
        "//xla:literal",
        "//xla/client:local_client",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "reshape_motion_test",
    srcs = ["reshape_motion_test.cc"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deep_graph_test",
    srcs = ["deep_graph_test.cc"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla/hlo/builder:xla_builder",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "literal_test_util_test",
    srcs = ["literal_test_util_test.cc"],
    deps = [
        ":literal_test_util",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "transfer_manager_test",
    srcs = ["transfer_manager_test.cc"],
    shard_count = 15,
    use_legacy_runtime = True,
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":xla_internal_test_main",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:generic_transfer_manager",
        "//xla/service:shaped_buffer",
        "//xla/service:stream_pool",
        "//xla/stream_executor:device_address_allocator",
        "//xla/tests:xla_test_backend_predicates",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
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
    tags = [
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":hlo_pjrt_test_base",
        ":hlo_runner_agnostic_reference_mixin",
        ":xla_internal_test_main",  # fixdeps: keep
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    # There is nothing backend specific in this test, so just pick an arbitrary backend.
    backends = ["cpu"],
    use_legacy_runtime = True,
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
        "@tsl//tsl/platform:ml_dtypes",
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
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "ptxas_bug_120501638_test",
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
    shard_count = 50,
    tags = [
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
    shard_count = 20,
    tags = [
        "optonly",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
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
        "@com_google_absl//absl/strings",
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
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
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
        "//xla/tsl/platform/default:cuda_build_defs_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
    ] + tests_build_defs_bzl_deps(),
)

xla_test(
    name = "atan2_test",
    srcs = ["atan2_test.cc"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
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
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        "//xla:literal_util",
        "//xla/hlo/builder:xla_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "two_plus_two_simple_test",
    srcs = ["two_plus_two_simple_test.cc"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":client_library_test_runner_mixin",
        ":hlo_pjrt_interpreter_reference_mixin",
        ":hlo_pjrt_test_base",
        ":xla_internal_test_main",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "aot_utils",
    testonly = True,
    srcs = ["aot_utils.cc"],
    hdrs = ["aot_utils.h"],
    deps = [
        "//xla/hlo/evaluator:caching_hlo_evaluator",
        "//xla/hlo/evaluator:hlo_evaluator_interface",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/interpreter:interpreter_client",
        "//xla/service:hlo_runner_pjrt",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xla_test_backend_predicates",
    testonly = True,
    srcs = ["xla_test_backend_predicates.cc"],
    hdrs = ["xla_test_backend_predicates.h"],
    tags = if_google([
        "avoid_dep",
        "alt_dep=" + xla_internal(
            ["tests:xla_test_backend_predicates"],
            [""],
        )[0],
    ]),
    visibility = if_google(
        xla_internal([
            "tests:__pkg__",
        ]) + [
            "//xla/tests/exhaustive:__pkg__",
        ],
        ["//visibility:public"],
    ),
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "hlo_test_base_with_mlir_context",
    testonly = True,
    hdrs = ["hlo_test_base_with_mlir_context.h"],
    deps = [
        ":hlo_test_base",
        "//xla/hlo/analysis:symbolic_expr",
        "@llvm-project//mlir:IR",
    ],
)
