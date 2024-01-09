# Description:
#
# This package contains shim library targets for the Async C package.
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    srcs = ["backend_async_kernel_interface.cc"],
    hdrs = ["backend_async_kernel_interface.h"],
    tflite_deps = [
        "//tensorflow/lite/async/c:async_kernel",
        "//tensorflow/lite/async/c:types",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "backend_async_kernel_interface_test",
    srcs = ["backend_async_kernel_interface_test.cc"],
    deps = [
        ":backend_async_kernel_interface",
        "//tensorflow/lite/async/c:types",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/async:async_kernel_internal",
        "//tensorflow/lite/core/async/testing:mock_async_kernel",
        "@com_google_googletest//:gtest_main",
    ],
)
