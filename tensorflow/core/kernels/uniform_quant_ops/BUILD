# Description: OpKernels for Uniform quant ops.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

# copybara:uncomment_begin(google-only)
# # Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
# copybara:uncomment_end
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "uniform_quantize_op.cc",
        "uniform_quantized_add_op.cc",
        "uniform_dequantize_op.cc",
        "uniform_requantize_op.cc",
        "uniform_quantized_dot_ops.cc",
        "uniform_quantized_convolution_ops.cc",
        "uniform_quantized_clip_by_value_op.cc",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

tf_kernel_library(
    name = "kernels",
    srcs = [
        "uniform_dequantize_op.cc",
        "uniform_quantize_op.cc",
        "uniform_quantized_add_op.cc",
        "uniform_quantized_clip_by_value_op.cc",
        "uniform_quantized_convolution_ops.cc",
        "uniform_quantized_dot_ops.cc",
        "uniform_requantize_op.cc",
    ],
    deps = [
        ":math_utils",
        ":tensor_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_params",
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
    name = "uniform_quantize_op_test",
    size = "small",
    srcs = ["uniform_quantize_op_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "uniform_requantize_op_test",
    size = "small",
    srcs = ["uniform_requantize_op_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "uniform_dequantize_op_test",
    size = "small",
    srcs = ["uniform_dequantize_op_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "uniform_quantized_dot_ops_test",
    size = "small",
    srcs = ["uniform_quantized_dot_ops_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "uniform_quantized_convolution_ops_test",
    size = "small",
    srcs = ["uniform_quantized_convolution_ops_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:protobuf",
    ],
)

tf_cc_test(
    name = "uniform_quantized_add_op_test",
    size = "small",
    srcs = ["uniform_quantized_add_op_test.cc"],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "uniform_quantized_clip_by_value_op_test",
    size = "small",
    srcs = ["uniform_quantized_clip_by_value_op_test.cc"],
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
