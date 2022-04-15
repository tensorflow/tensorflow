load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")

package(
    default_visibility = ["//tensorflow/distribute/experimental/rpc/kernels:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    deps = tf_grpc_cc_dependencies(),
    alwayslink = 1,
)
