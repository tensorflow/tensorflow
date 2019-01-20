# Stateless random ops

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

py_library(
    name = "stateless",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "stateless_random_ops_test",
    srcs = ["python/kernel_tests/stateless_random_ops_test.py"],
    additional_deps = [
        ":stateless",
        "//tensorflow/python:client_testlib",
    ],
)
