# Description:
#   Contains kernels for layers.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "bucketization_kernel",
    srcs = ["bucketization_kernel.cc"],
    deps = [
        "//:protobuf_headers",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_feature_cross_kernel",
    srcs = ["sparse_feature_cross_kernel.cc"],
    deps = [
        "//:protobuf_headers",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
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
