# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrappers_cc")

cc_library(
    name = "ops",
    srcs = ["framework/ops.cc"],
    hdrs = ["framework/ops.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "scope",
    srcs = ["framework/scope.cc"],
    hdrs = ["framework/scope.h"],
    deps = [
        ":ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "framework/scope_test.cc",
    deps = [
        ":ops",
        ":scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "const_op",
    srcs = ["ops/const_op.cc"],
    hdrs = ["ops/const_op.h"],
    deps = [
        ":ops",
        ":scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "ops/const_op_test.cc",
    deps = [
        ":const_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "grad_op_registry",
    srcs = ["framework/grad_op_registry.cc"],
    hdrs = ["framework/grad_op_registry.h"],
    deps = [
        ":ops",
        ":scope",
    ],
)

cc_library(
    name = "math_grad",
    srcs = ["gradients/math_grad.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":ops",
        ":scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "gradients/math_grad_test",
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":math_grad",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

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
    pkg = "//tensorflow/core",
)

tf_cc_test(
    name = "framework/cc_ops_test.cc",
    deps = [
        ":cc_ops",
        ":test_op",
        ":test_op_op_lib",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "sendrecv_ops",
    op_lib_names = [
        "sendrecv_ops",
    ],
    pkg = "//tensorflow/core",
)

cc_library(
    name = "cc_op_gen_main",
    srcs = [
        "framework/cc_op_gen.cc",
        "framework/cc_op_gen.h",
        "framework/cc_op_gen_main.cc",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "test_op_op_lib",
    srcs = ["framework/test_op.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_gen_op_wrappers_cc(
    name = "test_op",
    op_lib_names = [
        "test_op",
    ],
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
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
