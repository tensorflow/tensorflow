# Description:
#   Contains a Python wrapper for the StatSummarizer C++ class.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "stat_summarizer_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "stat_summarizer_test",
    size = "small",
    srcs = ["python/stat_summarizer_test.py"],
    additional_deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
    ],
)
