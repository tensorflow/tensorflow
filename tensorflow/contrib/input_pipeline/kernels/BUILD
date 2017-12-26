# Description:
#   Contains kernels for the input pipeline.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_library")

package(default_visibility = ["//tensorflow:__subpackages__"])

tf_cc_library(
    name = "input_pipeline_kernels",
    srcs = ["input_pipeline_kernels.cc"],
    is_external = True,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
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
