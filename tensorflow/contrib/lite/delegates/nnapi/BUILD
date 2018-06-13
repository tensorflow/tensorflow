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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/contrib/lite/nnapi:nnapi_lib",
    ],
)

tf_cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = ["nnapi_delegate_test.cc"],
    deps = [
        ":nnapi_delegate",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)
