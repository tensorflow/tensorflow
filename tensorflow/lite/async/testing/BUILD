# Test utilities for TFLite async execution.
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    tflite_deps = [
        "//tensorflow/lite/async:backend_async_kernel_interface",
        "//tensorflow/lite/async/c:types",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
    ],
)
