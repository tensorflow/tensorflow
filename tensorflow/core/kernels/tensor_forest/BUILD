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
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
	"//tensorflow/core:framework_headers_lib",
	"//third_party/eigen3",
    ],
)

cc_library(
    name = "resources",
    hdrs = ["resources.h", "evaluator.h"],
    srcs = ["resources.cc"],
    deps = [
        ":leaf_model",
        "//tensorflow/core/kernels/tensor_forest:tensor_forest_proto_cc",
	"//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "resource_ops",
    srcs = ["resource_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:tensor_forest_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "prediction_ops",
    srcs = ["prediction_ops.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:tensor_forest_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)
tf_kernel_library(
    name = "tensor_forest_ops",
    deps = [
        ":resource_ops",
        ":prediction_ops",
    ],
)
