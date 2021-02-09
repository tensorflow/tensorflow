# Description:
#   AST round-trip manipulation for Python.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "astunparse",
    srcs = [
        "astunparse/__init__.py",
        "astunparse/printer.py",
        "astunparse/unparser.py",
    ],
    srcs_version = "PY3",
)

genrule(
    name = "license",
    srcs = ["@astunparse_license"],
    outs = ["LICENSE"],
    cmd = "cp $< $@",
)
