# Description:
#   Contains kernels for evaluation metrics and summary statistics.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "set_kernels",
    srcs = ["set_kernels.cc"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

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
