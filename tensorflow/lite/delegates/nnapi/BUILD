package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)

tf_cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = ["nnapi_delegate_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)
