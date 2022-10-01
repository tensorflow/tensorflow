load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
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
        "//tensorflow/lite/c:common",
        "//third_party/eigen3",
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
        "@com_google_googletest//:gtest_main",
    ],
)
