package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_library",
    "cc_header_only_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)

cc_library(
    name = "ctc_cpu",
    srcs = ["test_cpu.cpp"],
    hdrs = ["test.h"],
    deps = ["//tensorflow/core/kernels/warp-ctc/include:libcpu", "//tensorflow/core/kernels/warp-ctc/src:src_cpu"],
)

tf_kernel_library(
    name = "ctc_gpu",
    gpu_srcs = ["test_gpu.h"],
    hdrs = ["test.h"],
    deps = ["//tensorflow/core/kernels/warp-ctc/include:libcpu", "//tensorflow/core/kernels/warp-ctc/src:src_gpu"],
    alwayslink = 1,
)
