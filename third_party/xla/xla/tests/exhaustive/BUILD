# Description:
#   Computationally expensive, exhaustive tests for XLA

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tests/exhaustive:build_defs.bzl", "exhaustive_xla_test")

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
    packages = [
        "//platforms/testing/tests/...",
    ],
)

cc_library(
    name = "exhaustive_op_test_utils",
    testonly = True,
    srcs = ["exhaustive_op_test_utils.cc"],
    hdrs = ["exhaustive_op_test_utils.h"],
    deps = [
        "//xla:bit_cast",
        "//xla:executable_run_options",
        "//xla:fp_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/service:shaped_buffer",
        "//xla/tests:client_library_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

filegroup(
    name = "exhaustive_unary_test_srcs",
    testonly = True,
    srcs = [
        "exhaustive_unary_test_definitions.h",
        "exhaustive_unary_test_functions.cc",
    ],
)

filegroup(
    name = "exhaustive_unary_test_f32_and_smaller_srcs",
    testonly = True,
    srcs = ["exhaustive_unary_test_f32_and_smaller_instantiation.cc"],
)

filegroup(
    name = "exhaustive_unary_test_f64_srcs",
    testonly = True,
    srcs = ["exhaustive_unary_test_f64_instantiation.cc"],
)

exhaustive_xla_test(
    name = "exhaustive_unary_test",
    timeout = "long",
    srcs = [
        "exhaustive_test_main.cc",
        ":exhaustive_unary_test_srcs",
    ],
    backends = [
        "gpu",
        "cpu",
    ],
    # Partition goal is to get under ~5 minute execution time without needlessly reserving hardware
    # for each shard.
    #
    # exhaustive_xla_test needs to have all partition names added to allow other build tools to
    # function.
    partitions = {
        "f32_and_smaller": [":exhaustive_unary_test_f32_and_smaller_srcs"],
        "f64": [":exhaustive_unary_test_f64_srcs"],
    },
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla:literal",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/client/lib:constants",
        "//xla/client/lib:math",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_unary_complex_test",
    timeout = "long",
    srcs = [
        "exhaustive_test_main.cc",
        "exhaustive_unary_complex_test.cc",
    ],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla:literal",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

filegroup(
    name = "exhaustive_binary_16_bit_test_srcs",
    srcs = [
        "exhaustive_binary_16_bit_test.cc",
    ],
)

xla_test(
    name = "exhaustive_binary_16_bit_test",
    timeout = "long",
    srcs = [
        "exhaustive_test_main.cc",
        ":exhaustive_binary_16_bit_test_srcs",
    ],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla:literal",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_binary_f32_f64_test",
    timeout = "long",
    srcs = [
        "exhaustive_binary_f32_f64_test.cc",
        "exhaustive_test_main.cc",
    ],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla:literal",
        "//xla/client:xla_builder",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)
