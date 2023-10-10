# This package contains shim library targets for the Async C package,
# that forward to the TF Lite C and C++ API targets.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "task",
    hdrs = ["task.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/async/c:task"],
)

cc_library_with_tflite(
    name = "types",
    hdrs = ["types.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/async/c:types"],
)

cc_library_with_tflite(
    name = "async_signature_runner",
    hdrs = ["async_signature_runner.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/async/c:async_signature_runner"],
)

cc_library_with_tflite(
    name = "async_kernel",
    hdrs = ["async_kernel.h"],
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/async/c:async_kernel"],
)
