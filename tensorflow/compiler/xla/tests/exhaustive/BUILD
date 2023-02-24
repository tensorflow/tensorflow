# Description:
#   Computationally expensive, exhaustive tests for XLA

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

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
        "//tensorflow/compiler/xla:bit_cast",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "exhaustive_unary_test_f32_or_smaller",
    srcs = ["exhaustive_unary_test_f32_or_smaller.cc"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
    ],
)

xla_test(
    name = "exhaustive_unary_test_f64",
    srcs = ["exhaustive_unary_test_f64.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
        # TODO(b/151340488): Timed out on 2020-03-18.
        "nozapfhahn",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
    ],
)

xla_test(
    name = "exhaustive_unary_test_complex",
    srcs = ["exhaustive_unary_test_complex.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
    ],
)

xla_test(
    name = "exhaustive_binary_16_bit_test",
    srcs = ["exhaustive_binary_16_bit_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)

xla_test(
    name = "exhaustive_binary_test_f32_f64",
    srcs = ["exhaustive_binary_test_f32_f64.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)
