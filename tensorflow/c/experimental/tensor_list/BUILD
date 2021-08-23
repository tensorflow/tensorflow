# Description:
# TensorList C APIs.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],
)

cc_library(
    name = "tensor_list_c_hdrs",
    hdrs = ["tensor_list.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "tensor_list",
    srcs = [
        "tensor_list.cc",
    ],
    hdrs = [
        "tensor_list.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:tensor_list_hdrs",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_test(
    name = "tensor_list_test",
    size = "small",
    srcs = ["tensor_list_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    tags = ["noasan"],
    deps = [
        ":tensor_list",
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:tensor_list",
        "@com_google_absl//absl/strings",
    ],
)
