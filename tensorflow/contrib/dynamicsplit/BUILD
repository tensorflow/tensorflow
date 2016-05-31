# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "dynamicsplit_py",
    srcs = [
        "__init__.py",
        "dynamicsplit_ops.py",
    ],
    srcs_version = "PY2AND3",
)


py_test(
    name = "dynamicsplit_op_test",
    size = "small",
    srcs = ["dynamicsplit_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dynamicsplit_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
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
