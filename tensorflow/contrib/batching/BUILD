# Description: Batching scheduling library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

py_library(
    name = "batch_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:batch_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:util",
    ],
)

cc_library(
    name = "batch_ops_kernels",
    deps = [
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
