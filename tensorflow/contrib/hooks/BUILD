# Description:
#   Contains `SessionRunHook`s for use with `MonitoredSession` and the
#   wrappers around it.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "hooks",
    srcs = [
        "__init__.py",
        "python/training/__init__.py",
        "python/training/profiler_hook.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)
