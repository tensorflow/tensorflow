# Description:
#   functools32 provides a backport of the functools module for Python 2.

licenses(["notice"])  # Python 2.0

exports_files(["LICENSE"])

py_library(
    name = "functools32",
    srcs = [
        "functools32/__init__.py",
        "functools32/_dummy_thread32.py",
        "functools32/functools32.py",
        "functools32/reprlib32.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
