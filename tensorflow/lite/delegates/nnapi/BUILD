package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "nnapi_delegate",
    srcs = select({
        "//tensorflow:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
        ],
    }),
    hdrs = ["nnapi_delegate.h"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
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
        "no_windows",
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
