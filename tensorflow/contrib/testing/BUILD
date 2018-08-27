# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "testing_py",
    srcs = [
        "__init__.py",
        "python/framework/fake_summary_writer.py",
        "python/framework/util_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)
