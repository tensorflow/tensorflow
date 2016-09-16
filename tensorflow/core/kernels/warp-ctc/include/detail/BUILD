package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_library",
    "cc_header_only_library",
)

cc_library(
    name = "libcpu",
    srcs = ["cpu_ctc.h", "ctc_helper.h", "hostdevice.h"],
    linkopts = ["-lgomp"],
)

cc_library(
    name = "libgpu",
    srcs = glob(["*.h"]),
    linkopts = ["-lgomp"],
)
