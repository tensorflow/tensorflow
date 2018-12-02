# Description:
#   Common functionality for TensorFlow tooling

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow:__subpackages__"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "public_api",
    srcs = ["public_api.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:util"],
)

py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":public_api",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "traverse",
    srcs = ["traverse.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:util"],
)

py_test(
    name = "traverse_test",
    srcs = ["traverse_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_module1",
        ":test_module2",
        ":traverse",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "test_module1",
    srcs = ["test_module1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_module2",
    ],
)

py_library(
    name = "test_module2",
    srcs = ["test_module2.py"],
    srcs_version = "PY2AND3",
)
