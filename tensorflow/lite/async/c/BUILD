# This package contains shim library targets for the Async C package,
# that forward to the TF Lite C and C++ API targets.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")

cc_library(
    name = "task",
    hdrs = ["task.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/async/c:task"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/async/c:types"],
)

cc_library(
    name = "async_signature_runner",
    hdrs = ["async_signature_runner.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/async/c:async_signature_runner"],
)
