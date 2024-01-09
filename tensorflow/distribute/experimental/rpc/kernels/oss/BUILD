load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/distribute/experimental/rpc/kernels:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    deps = tf_grpc_cc_dependencies(),
    alwayslink = 1,
)
