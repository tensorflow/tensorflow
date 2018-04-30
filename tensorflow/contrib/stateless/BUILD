# Stateless random ops

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

tf_gen_op_wrapper_py(
    name = "stateless_random_ops",
    out = "gen_stateless_random_ops.py",  # cmake chokes without this
    deps = ["//tensorflow/core:stateless_random_ops_op_lib"],
)

py_library(
    name = "stateless",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":stateless_random_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "stateless_random_ops_test",
    srcs = ["python/kernel_tests/stateless_random_ops_test.py"],
    additional_deps = [
        ":stateless",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
    ],
)
