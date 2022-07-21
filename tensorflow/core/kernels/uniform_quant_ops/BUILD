# Description: OpKernels for Uniform quant ops.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

# copybara:uncomment_begin(google-only)
# # Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
# copybara:uncomment_end
load("//tensorflow:tensorflow.bzl", "filegroup")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")  # buildifier: disable=same-origin-load

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

filegroup(
    name = "portable_all_op_kernels_headers",
    srcs = [
        "math_utils.h",
        "tensor_utils.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        ":portable_all_op_kernels_headers",
    ] + [
        "math_utils.cc",
        "tensor_utils.cc",
        "uniform_quantized_dot_ops.cc",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

tf_kernel_library(
    name = "kernels",
    srcs = [
        "uniform_quantized_dot_ops.cc",
    ],
    deps = [
        ":math_utils",
        ":tensor_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "math_utils",
    srcs = ["math_utils.cc"],
    hdrs = ["math_utils.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.cc"],
    hdrs = ["tensor_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "uniform_quantized_dot_ops_test",
    size = "small",
    srcs = ["uniform_quantized_dot_ops_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "math_utils_test",
    srcs = ["math_utils_test.cc"],
    deps = [
        ":math_utils",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    deps = [
        ":tensor_utils",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
