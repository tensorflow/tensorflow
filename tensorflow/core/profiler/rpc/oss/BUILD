load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")

package(
    default_visibility = [
        "//tensorflow/core/profiler:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grpc",
    srcs = [
        "grpc.cc",
        "//tensorflow/core/profiler/rpc:grpc.h",
    ],
    deps = [
        tf_grpc_cc_dependency(),
    ],
    alwayslink = True,
)

exports_files(
    [
        "grpc.cc",
    ],
    visibility = ["//tensorflow/core/profiler/rpc:__subpackages__"],
)
