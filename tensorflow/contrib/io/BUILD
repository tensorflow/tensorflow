# Description:
# Contributed input-output op kernel implementations for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "py_test",
)

tf_custom_op_library(
    name = "python/ops/_io_ops.so",
    srcs = [
        "ops/io_ops.cc",
        "kernels/zmq_op.cc",
        ],
    deps = ["@libzmq//:libzmq"],
)

tf_gen_op_libs(
    op_lib_names = ["io_ops"],
)

tf_gen_op_wrapper_py(
    name = "io_ops",
    deps = [":io_ops_op_lib"],
)

tf_kernel_library(
    name = "io_ops_kernels",
    deps = [
        "//tensorflow/contrib/io/kernels:io_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "io_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    dso = [":python/ops/_io_ops.so"],
    kernels = [
        ":io_ops_kernels",
        ":io_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":io_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "io_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/io_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":io_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)
