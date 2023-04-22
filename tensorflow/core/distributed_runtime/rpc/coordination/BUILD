load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_coordination_client",
    hdrs = ["grpc_coordination_client.h"],
    deps = [
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "grpc_coordination_client_impl",
    srcs = ["grpc_coordination_client.cc"],
    deps = [
        ":grpc_coordination_client",
        tf_grpc_cc_dependency(),
    ],
)
