# TODO(suharshs): Write quantize_weights tests that use small exportable files.
# Then we can remove this file.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "quantize_weights",
    srcs = ["quantize_weights.cc"],
    hdrs = ["quantize_weights.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels/internal:tensor_utils",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/core:tflite_portable_logging",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)
