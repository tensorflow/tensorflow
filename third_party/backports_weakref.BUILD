# Description:
#   Backport of new features in Python's weakref module.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Python 2.0

py_library(
    name = "org_python_pypi_backports_weakref",
    srcs = [
        "backports/__init__.py",
        "backports/weakref.py",
    ],
    srcs_version = "PY3",
)

genrule(
    name = "license",
    srcs = ["@org_python_license"],
    outs = ["LICENSE"],
    cmd = "cp $< $@",
)
