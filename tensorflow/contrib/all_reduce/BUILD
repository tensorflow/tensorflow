# Description:
#   All-reduce implementations.
#   APIs are subject to change.  Eventually to be replaced by equivalent
#   functionality within TensorFlow core.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "all_reduce_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_reduce",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "all_reduce",
    srcs = [
        "python/all_reduce.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/distribute:all_reduce",
    ],
)
