# Test utilities for TFLite async execution.

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
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/async:async_kernel_internal",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:utils",
    ],
)

cc_library(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    deps = [
        "//tensorflow/lite/core/async:backend_async_kernel_interface",
        "//tensorflow/lite/core/async/c:types",
        "@com_google_googletest//:gtest",
    ],
)
