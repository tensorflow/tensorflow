# Description:
#   Contains kernels for cudnn batchnorm

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_kernel_library(
    name = "cudnn_batchnorm",
    srcs = ["cudnn_batchnorm.cc"],
    deps = [
        "//google/protobuf",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:conv_2d",
        "//third_party/eigen3",
    ],
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
