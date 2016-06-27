package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "is_ci",
    values = {"define": "CI=true"}
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_libraries",
    "tf_kernel_library",
    "cc_header_only_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)

tf_kernel_library(
    name = "warpctc",
    srcs = ["test.h"],
    gpu_srcs = ["test_gpu.h"],
    deps = select({
        ":is_ci": ["//tensorflow/core/kernels/warp-ctc/src:src_cpu"],
        "//conditions:default": ["//tensorflow/core/kernels/warp-ctc/src:src_gpu"],
    }) + ["//tensorflow/core/kernels/warp-ctc/include:libcpu"],
    alwayslink = 1,
)
