# Description:
# Platform-specific build configurations.

load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.bzl", "set_external_visibility", "tsl_copts")

package(default_visibility = set_external_visibility(["//tensorflow/tsl:internal"]))

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "gtest",
    testonly = 1,
    copts = tsl_copts(),
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tensorflow_platform_specific",
    copts = tsl_copts(),
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "_empty_lib",
    visibility = ["//visibility:private"],
)

# Dummy stream executor cuda plugins.
cc_library(
    name = "cublas_plugin",
    srcs = [],
)

cc_library(
    name = "cufft_plugin",
    srcs = [],
)

cc_library(
    name = "cudnn_plugin",
    srcs = [],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on platformlib.
cc_library(
    name = "proto_parsing",
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/protobuf:protos_all_cc",
    ],
)

# Minimal lib to be used by tensorflow/core:framework_lite.
# This provides minimal support for writing operator implementations (kernels),
# and excludes anything that can bloat binary size if used.
cc_library(
    name = "minimal",
    srcs = [],
    copts = tsl_copts(),
)

cc_library(
    name = "gif",
    copts = tsl_copts(),
    deps = [
        "@gif",
    ],
)

cc_library(
    name = "jpeg",
    copts = tsl_copts(),
    deps = [
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "png",
    copts = tsl_copts(),
    deps = [
        "@png",
        "@zlib",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "//tensorflow/tsl:macos": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)
