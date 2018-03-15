# Description: Utilities to aid testing.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    hdrs = ["fake_clock_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels/batching_util:fake_clock_env",
    ],
)
