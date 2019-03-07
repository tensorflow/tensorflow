# Description:
#   OpKernels for tensor forest ops.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "tensor_forest_proto",
    srcs = ["tensor_forest.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "resources",
    srcs = ["resources.cc"],
    hdrs = ["resources.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
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
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
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
    name = "train_ops",
    srcs = ["train_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_forest_ops_op_lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
    ],
)

tf_kernel_library(
    name = "tensor_forest_ops",
    deps = [
        ":prediction_ops",
        ":resource_ops",
        ":train_ops",
    ],
)
