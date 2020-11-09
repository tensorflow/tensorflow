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
    name = "experimental",
    deps = [
        ":risc_add_op",
    ],
)
