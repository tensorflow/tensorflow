# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "lookup_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/feature_column:lookup_ops",
    ],
)

py_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lookup_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
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
