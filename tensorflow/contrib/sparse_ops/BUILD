# Description
#   This module provides several sparse operations. They might be helpful if you are interested in graph neural networks.


package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")


tf_custom_op_library(
    name = "python/ops/_sparse_ops.so",
    srcs = [
        "kernels/sparse_ops.cc",
        "ops/sparse_ops.cc",
    ],     
    deps = [
    ],
)

cc_library(
    name = "sparse_ops_lib",
    srcs = [
        "kernels/sparse_ops.cc",
        "ops/sparse_ops.cc"
    ],
    deps = [
        "//tensorflow/core/kernels:sparse",
        "//tensorflow/core:framework",
    ]
)

tf_kernel_library(
    name = "sparse_ops_kernels",
    deps = [
        ":sparse_ops_lib"
    ],
)


tf_gen_op_libs(
    op_lib_names = ["sparse_ops"],
)

tf_gen_op_wrapper_py(
    name = "sparse_ops",
    deps = [":sparse_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "sparse_ops_py",
    srcs = glob(["python/ops/*.py"]),
    dso = [":python/ops/_sparse_ops.so"],
    kernels = [":sparse_ops_kernels"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":sparse_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:framework_ops",
    ],
)

tf_py_test(
    name = "sparse_ops_test",
    size = "small",
    srcs = ["python/pytest/sparse_ops_test.py"],
    additional_deps = [
        ":sparse_ops_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_cc_test(
    name = "sparse_ops_test_cc",
    size = "small",
    srcs = [
        "kernels/sparse_ops_test.cc",
        # "ops/sparse_ops_test.cc",
    ],
    deps = [
        ":sparse_ops_lib",
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
