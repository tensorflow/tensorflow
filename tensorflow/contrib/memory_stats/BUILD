# Description:
#   Ops that get statistics on memory allocators.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_memory_stats_ops.so",
    srcs = [
        "kernels/memory_stats_ops.cc",
        "ops/memory_stats_ops.cc",
    ],
)

tf_kernel_library(
    name = "memory_stats_kernels",
    srcs = [
        "kernels/memory_stats_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["memory_stats_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "memory_stats_ops",
    deps = [":memory_stats_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "memory_stats_py",
    srcs = [
        "__init__.py",
        "python/ops/memory_stats_ops.py",
    ],
    dso = [
        ":python/ops/_memory_stats_ops.so",
    ],
    kernels = [
        ":memory_stats_kernels",
        ":memory_stats_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":memory_stats_ops",
        "//tensorflow/contrib/util:util_py",
    ],
)

cuda_py_test(
    name = "memory_stats_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/memory_stats_ops_test.py"],
    additional_deps = [
        ":memory_stats_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
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
