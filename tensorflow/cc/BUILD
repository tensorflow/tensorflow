# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrappers_cc")

cc_library(
    name = "cc_op_gen_main",
    srcs = [
        "ops/cc_op_gen.cc",
        "ops/cc_op_gen_main.cc",
    ],
    hdrs = ["ops/cc_op_gen.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
    ],
)

# Generates a library that contains C++ wrappers for ops.
tf_gen_op_wrappers_cc(
    name = "cc_ops",
    op_lib_names = [
        "array_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "data_flow_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "logging_ops",
        "math_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
        "sendrecv_ops",
        "sparse_ops",
        "state_ops",
        "string_ops",
        "training_ops",
        "user_ops",
    ],
    other_hdrs = [
        "ops/const_op.h",
        "ops/standard_ops.h",
    ],
    other_srcs = [
        "ops/const_op.cc",
    ] + glob(["ops/*_grad.cc"]),
    pkg = "//tensorflow/core",
)

cc_binary(
    name = "tutorials_example_trainer",
    srcs = ["tutorials/example_trainer.cc"],
    copts = tf_copts(),
    linkopts = [
        "-lpthread",
        "-lm",
    ],
    deps = [
        ":cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:tensorflow",
    ],
)

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
