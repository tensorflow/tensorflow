# Description:
#   OpKernels for tensor forest ops.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

cc_library(
    name = "resources",
    srcs = ["resources.cc"],
    hdrs = ["resources.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

tf_kernel_library(
    name = "resource_ops",
    srcs = ["resource_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

tf_kernel_library(
    name = "prediction_ops",
    srcs = ["prediction_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

tf_kernel_library(
    name = "tensor_forest_ops",
    deps = [
        ":prediction_ops",
        ":resource_ops",
    ],
)
