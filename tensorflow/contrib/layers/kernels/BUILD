# Description:
#   Contains kernels for layers.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_library")

package(default_visibility = ["//tensorflow:__subpackages__"])

tf_cc_library(
    name = "sparse_feature_cross_kernel",
    srcs = ["sparse_feature_cross_kernel.cc"],
    is_external = True,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
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
