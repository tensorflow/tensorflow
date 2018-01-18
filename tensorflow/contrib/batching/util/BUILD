# Description: Utilities.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/google_*",
        ],
    ),
)

cc_library(
    name = "periodic_function_dynamic",
    hdrs = ["periodic_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "periodic_function",
    visibility = ["//visibility:public"],
    deps = [
        ":periodic_function_dynamic",
        "//tensorflow/core/kernels/batching_util:periodic_function",
    ],
)
