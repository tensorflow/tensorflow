# TF-RISC

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_kernel_library(
    name = "risc_add_op",
    srcs = ["risc_add_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_broadcast_op",
    srcs = ["risc_broadcast_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_concat_op",
    srcs = ["risc_concat_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_conv_op",
    srcs = ["risc_conv_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_dot_op",
    srcs = ["risc_dot_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_max_op",
    srcs = ["risc_max_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_pad_op",
    srcs = ["risc_pad_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_pool_op",
    srcs = ["risc_pool_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_reshape_op",
    srcs = ["risc_reshape_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_shape_op",
    srcs = ["risc_shape_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_slice_op",
    srcs = ["risc_slice_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "experimental",
    deps = [
        ":risc_add_op",
        ":risc_broadcast_op",
        ":risc_conv_op",
        ":risc_dot_op",
        ":risc_max_op",
        ":risc_pad_op",
        ":risc_pool_op",
        ":risc_reshape_op",
        ":risc_shape_op",
        ":risc_slice_op",
    ],
)
