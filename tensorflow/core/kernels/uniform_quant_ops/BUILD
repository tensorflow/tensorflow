# Description: OpKernels for Uniform quant ops.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

# For mobile.
exports_files([
    "uniform_quantized_dot_ops.cc",
])

tf_kernel_library(
    name = "kernels",
    srcs = [
        "uniform_quantized_dot_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
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
