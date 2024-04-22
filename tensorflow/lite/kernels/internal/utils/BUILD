load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "sparsity_format_converter",
    srcs = ["sparsity_format_converter.cc"],
    hdrs = ["sparsity_format_converter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "sparsity_format_converter_test",
    srcs = ["sparsity_format_converter_test.cc"],
    data = ["//tensorflow/lite:testdata/sparse_tensor.bin"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":sparsity_format_converter",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "@com_google_googletest//:gtest_main",
    ],
)
