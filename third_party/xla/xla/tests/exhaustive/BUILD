# Description:
#   Computationally expensive, exhaustive tests for XLA

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")

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
    tags = ["no_pip"],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
        "//xla:bit_cast",
        "//xla:executable_run_options",
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
    ],
)

filegroup(
    name = "exhaustive_unary_f32_or_smaller_test_srcs",
    testonly = True,
    srcs = ["exhaustive_unary_f32_or_smaller_test.cc"],
)

xla_test(
    name = "exhaustive_unary_f32_or_smaller_test",
    srcs = [
        "exhaustive_test_main.cc",
        ":exhaustive_unary_f32_or_smaller_test_srcs",
    ],
    backends = [
        "gpu",
        "cpu",
    ],
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
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client/lib:constants",
        "//xla/client/lib:math",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_unary_f64_test",
    srcs = [
        "exhaustive_test_main.cc",
        "exhaustive_unary_f64_test.cc",
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
        # TODO(b/151340488): Timed out on 2020-03-18.
        "nozapfhahn",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client/lib:constants",
        "//xla/client/lib:math",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_unary_complex_test",
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
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_binary_16_bit_test",
    srcs = [
        "exhaustive_binary_16_bit_test.cc",
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
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_binary_f32_f64_test",
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
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)
