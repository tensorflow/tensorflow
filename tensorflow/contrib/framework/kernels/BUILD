#OpKernels for hacking initialization check

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

cc_library(
    name = "all_kernels",
    deps = [
        ":zero_initializer_op",
        "@protobuf//:protobuf",
    ],
)

tf_kernel_library(
    name = "zero_initializer_op",
    srcs = [
        "zero_initializer_op.cc",
        "zero_initializer_op.h",
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
