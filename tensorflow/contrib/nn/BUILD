# Description:
#   Contains deprecated ops to calculate cross entropy.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "nn_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/ops/alpha_dropout.py",
        "python/ops/cross_entropy.py",
        "python/ops/sampling_ops.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "sampling_ops_test",
    size = "small",
    srcs = ["python/ops/sampling_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nn_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
    ],
)

py_test(
    name = "alpha_dropout_test",
    size = "small",
    srcs = ["python/ops/alpha_dropout_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
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
)
