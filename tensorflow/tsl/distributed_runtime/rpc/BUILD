# Description:
#   RPC communication interfaces and implementations for TensorFlow.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "async_service_interface",
    srcs = [],
    hdrs = ["async_service_interface.h"],
    deps = [],
)

cc_library(
    name = "grpc_call",
    srcs = [],
    hdrs = ["grpc_call.h"],
    deps = [
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:refcount",
    ] + tf_grpc_cc_dependencies(),
)
