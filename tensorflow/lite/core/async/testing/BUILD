# Test utilities for TFLite async execution.
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_backend",
    testonly = 1,
    srcs = ["test_backend.cc"],
    hdrs = ["test_backend.h"],
    deps = [
        "//tensorflow/lite:array",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/core/async:async_kernel_internal",
        "//tensorflow/lite/core/async/c:types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:utils",
    ],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    deprecation = "Use //tensorflow/lite/async/testing:mock_async_kernel instead.",
    tflite_deps = [
        "//tensorflow/lite/async/testing:mock_async_kernel",
    ],
)
