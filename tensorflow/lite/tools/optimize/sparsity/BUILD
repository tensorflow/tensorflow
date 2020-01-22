load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "format_converter",
    srcs = ["format_converter.cc"],
    hdrs = ["format_converter.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_test(
    name = "format_converter_test",
    srcs = ["format_converter_test.cc"],
    data = ["//tensorflow/lite:testdata/sparse_tensor.bin"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":format_converter",
        "//tensorflow/lite:framework",
        "@com_google_googletest//:gtest_main",
    ],
)
