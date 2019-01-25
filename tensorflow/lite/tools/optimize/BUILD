# TODO(suharshs): Write quantize_weights tests that use small exportable files.
# Then we can remove this file.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_test(
    name = "quantizatin_utils_test",
    srcs = ["quantization_utils_test.cc"],
    copts = ["-Wall"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = ["quantize_weights.cc"],
    hdrs = ["quantize_weights.h"],
    deps = [
        ":quantization_utils",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)
