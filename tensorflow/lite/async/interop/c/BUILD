# This package contains shim library targets for the Async interop package,
# that forward to the TF Lite C and C++ API targets.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")

cc_library(
    name = "attribute_map",
    hdrs = ["attribute_map.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/async/interop/c:attribute_map"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/async/interop/c:types"],
)
