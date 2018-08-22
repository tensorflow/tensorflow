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
    visibility = ["//visibility:public"],
)

cc_library(
    name = "leaf_model",
    srcs = ["leaf_model.cc"],
    hdrs = ["leaf_model.h"],
    deps = [
	"//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
    ],
)

cc_library(
    name = "evaluator",
    hdrs = ["evaluator.h"],
    srcs = ["evaluator.cc"],
    deps = [
        ":leaf_model",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
	"//third_party/eigen3",
    ],
)

cc_library(
    name = "resources",
    hdrs = ["resources.h"],
    srcs = ["resources.cc"],
    deps = [
        ":leaf_model",
        ":evaluator",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
	"//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "resource_ops",
    srcs = ["resource_ops.cc"],
    deps = [
        ":resources",
        ":leaf_model",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_forest_ops_op_lib",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
    ],
)

tf_kernel_library(
    name = "prediction_ops",
    srcs = ["prediction_ops.cc"],
    deps = [
        ":resources",
        ":leaf_model",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_forest_ops_op_lib",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
    ],
)
tf_kernel_library(
    name = "tensor_forest_ops",
    deps = [
        ":resource_ops",
        ":prediction_ops",
    ],
)
