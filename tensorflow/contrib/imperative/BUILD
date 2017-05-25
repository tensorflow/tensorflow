# Description:
#   Imperative mode for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "imperative",
    srcs = [
        "__init__.py",
        "imperative_graph.py",
        "imperative_mode.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_graph",
        ":imperative_mode",
    ],
)

py_library(
    name = "imperative_graph",
    srcs = ["imperative_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "imperative_mode",
    srcs = ["imperative_mode.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_graph",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
    ],
)

cuda_py_test(
    name = "imperative_test",
    size = "small",
    srcs = ["imperative_test.py"],
    additional_deps = [
        ":imperative_mode",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python:framework_test_lib",
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
