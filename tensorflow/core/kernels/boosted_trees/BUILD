# Description:
#   OpKernels for boosted trees ops.

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
    name = "boosted_trees_proto",
    srcs = ["boosted_trees.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_kernel_library(
    name = "prediction_ops",
    srcs = ["prediction_ops.cc"],
    deps = [
        ":resource_ops",
        ":resources",
        "//tensorflow/core:boosted_trees_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

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
        "//tensorflow/core:boosted_trees_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

tf_kernel_library(
    name = "stats_ops",
    srcs = ["stats_ops.cc"],
    deps = [
        "//tensorflow/core:boosted_trees_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "training_ops",
    srcs = ["training_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:boosted_trees_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_cc",
    ],
)

tf_kernel_library(
    name = "boosted_trees_ops",
    deps = [
        ":prediction_ops",
        ":resource_ops",
        ":stats_ops",
        ":training_ops",
    ],
)
