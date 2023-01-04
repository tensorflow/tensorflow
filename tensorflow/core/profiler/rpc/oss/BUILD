load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc",
    srcs = [
        "grpc.cc",
        "//tensorflow/core/profiler/rpc:grpc.h",
    ],
    copts = tf_profiler_copts(),
    deps = tf_grpc_cc_dependencies(),
    alwayslink = True,
)

exports_files(
    [
        "grpc.cc",
    ],
    visibility = ["//tensorflow/core/profiler/rpc:__subpackages__"],
)
