# Description:
#   A Cudnn RNN wrapper.
#   APIs are meant to change over time.
package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

tf_custom_op_library(
    name = "python/ops/_cudnn_rnn_ops.so",
    srcs = [
        "kernels/cudnn_rnn_ops.cc",
        "ops/cudnn_rnn_ops.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_kernel_library(
    name = "cudnn_rnn_kernels",
    srcs = ["kernels/cudnn_rnn_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//third_party/eigen3",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["cudnn_rnn_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "cudnn_rnn_ops",
    deps = [":cudnn_rnn_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "cudnn_rnn_py",
    srcs = [
        "__init__.py",
        "python/ops/cudnn_rnn_ops.py",
    ],
    dso = [
        ":python/ops/_cudnn_rnn_ops.so",
    ],
    kernels = [
        ":cudnn_rnn_kernels",
        ":cudnn_rnn_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cudnn_rnn_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
    ],
)

cuda_py_test(
    name = "cudnn_rnn_ops_test",
    size = "large",
    srcs = ["python/kernel_tests/cudnn_rnn_ops_test.py"],
    additional_deps = [
        ":cudnn_rnn_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python/ops/losses:losses",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
    shard_count = 6,
    tags = [
        "manual",
        "requires_cudnn5",
    ],
)

cuda_py_test(
    name = "cudnn_rnn_ops_benchmark",
    size = "large",
    srcs = ["python/kernel_tests/cudnn_rnn_ops_benchmark.py"],
    additional_deps = [
        ":cudnn_rnn_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    tags = [
        "manual",
        "noasan",  # http://b/62067814
        "nomsan",
        "notsan",
        "requires_cudnn5",
    ],
)

tf_cc_test(
    name = "cudnn_rnn_ops_test_cc",
    size = "small",
    srcs = [
        "ops/cudnn_rnn_ops_test.cc",
    ],
    deps = [
        ":cudnn_rnn_ops_op_lib",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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
