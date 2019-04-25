package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "nnapi_delegate",
    srcs = ["nnapi_delegate.cc"],
    hdrs = ["nnapi_delegate.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/nnapi:nnapi_implementation",
    ],
)

cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = ["nnapi_delegate_test.cc"],
    tags = [
        # TODO(b/122987564): Enable on Android after resolving API 27 failures.
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tflite_portable_test_suite()
