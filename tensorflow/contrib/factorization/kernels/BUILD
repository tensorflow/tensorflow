# OpKernels for data factorization and clustering.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "all_kernels",
    deps = [
        ":masked_matmul_ops",
        ":wals_solver_ops",
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "wals_solver_ops",
    srcs = ["wals_solver_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "masked_matmul_ops",
    srcs = ["masked_matmul_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)
