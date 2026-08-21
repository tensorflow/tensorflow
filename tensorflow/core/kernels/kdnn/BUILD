load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//third_party/KDNN:build_defs.bzl", "if_enable_kdnn", "kdnn_deps")

# KDNN kernels - support for KDNN optimized operators
# Currently supports: sigmoid
# Note: KDNN is mutually exclusive with MKL/oneDNN
tf_kernel_library(
    name = "kdnn_kernels",
    visibility = ["//visibility:public"],
    srcs = if_enable_kdnn(["kdnn_sigmoid_op.cc"]),
    hdrs = if_enable_kdnn(["kdnn_unary_op.h"]),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:fill_functor",
        "//tensorflow/core:lib",
    ] + kdnn_deps(),
)
