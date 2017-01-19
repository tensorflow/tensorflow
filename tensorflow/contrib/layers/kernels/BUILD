# Description:
#   Contains kernels for layers.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
cc_library(
    name = "bucketization_kernel",
    srcs = ["bucketization_kernel.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_feature_cross_kernel",
    srcs = ["sparse_feature_cross_kernel.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "layer_norm_fused_kernel",
    copts = tf_copts(),
    srcs = [
        "layer_norm_fused_op.cc",
        "layer_norm_fused_grad_op.cc",
        "layer_norm_fused_op_gpu.cu.cc",
        "layer_norm_fused_op.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
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
