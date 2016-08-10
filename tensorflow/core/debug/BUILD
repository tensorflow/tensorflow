# Description:
# TensorFlow Debugger (tfdbg).
#
# Public Android targets:
# filegroup ":android_srcs" - Debugger source files for Android.

package(
    default_visibility = ["//tensorflow:internal"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cc_test",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

tf_cuda_library(
    name = "debug_gateway_internal",
    srcs = ["debug_gateway.cc"],
    hdrs = ["debug_gateway.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_tracer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_graph_utils",
    srcs = ["debug_graph_utils.cc"],
    hdrs = ["debug_graph_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_io_utils",
    srcs = ["debug_io_utils.cc"],
    hdrs = ["debug_io_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cc_test_gpu(
    name = "debug_gateway_test",
    size = "small",
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":debug_gateway_internal",
        ":debug_graph_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:debug_ops",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "debug_io_utils_test",
    size = "small",
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":debug_io_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

filegroup(
    name = "android_srcs",
    srcs = [
        "debug_graph_utils.cc",
        "debug_graph_utils.h",
    ],
    visibility = ["//visibility:public"],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
