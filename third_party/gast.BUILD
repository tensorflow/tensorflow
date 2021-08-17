# Description:
#   Python AST that abstracts the underlying Python version.

licenses(["notice"])  # BSD 3-clause

exports_files(["PKG-INFO"])

py_library(
    name = "gast",
    srcs = [
        "gast/__init__.py",
        "gast/ast2.py",
        "gast/ast3.py",
        "gast/astn.py",
        "gast/gast.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
