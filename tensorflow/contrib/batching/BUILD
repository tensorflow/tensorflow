# Description: Batching scheduling library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
)

py_library(
    name = "batch_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:batch_ops",
        "//tensorflow/python:batch_ops_gen",
    ],
)

cc_library(
    name = "batch_ops_kernels",
    deps = [
        "//tensorflow/core:batch_ops_op_lib",
        "//tensorflow/core/kernels:batch_kernels",
    ],
    alwayslink = 1,
)

py_test(
    name = "batch_ops_test",
    size = "small",
    srcs = ["python/ops/batch_ops_test.py"],
    shard_count = 5,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "nomac",
    ],
    deps = [
        ":batch_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:script_ops",
    ],
)
