# Description: Utilities to aid testing.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

tf_cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)
