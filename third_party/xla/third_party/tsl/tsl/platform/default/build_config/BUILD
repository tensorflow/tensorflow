# Description:
# Platform-specific build configurations.

load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl:tsl.bzl", "set_external_visibility", "tsl_copts")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(
    ["LICENSE"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest",
    testonly = 1,
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tensorflow_platform_specific",
    copts = tsl_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "_empty_lib",
    visibility = ["//visibility:public"],
)

# Dummy stream executor cuda plugins.
cc_library(
    name = "cublas_plugin",
    srcs = [],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufft_plugin",
    srcs = [],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_plugin",
    srcs = [],
    visibility = ["//visibility:public"],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on platformlib.
cc_library(
    name = "proto_parsing",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/protobuf:protos_all_cc",
    ],
)

# Minimal lib to be used by tensorflow/core:framework_lite.
# This provides minimal support for writing operator implementations (kernels),
# and excludes anything that can bloat binary size if used.
cc_library(
    name = "minimal",
    srcs = [],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gif",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@gif",
    ],
)

cc_library(
    name = "jpeg",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "png",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@png",
        "@zlib",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = [
        "-Wl,-rpath,../local_config_cuda/cuda/lib64",
        "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)
