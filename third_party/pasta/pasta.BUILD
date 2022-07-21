# Description:
#   AST-based python refactoring.
load("//:build_defs.bzl", "copy_srcs")

licenses(["notice"])  # Apache2

exports_files(["LICENSE"])

py_library(
    name = "pasta",
    srcs = copy_srcs([
        "__init__.py",
        "augment/__init__.py",
        "augment/errors.py",
        "augment/import_utils.py",
        "augment/inline.py",
        "augment/rename.py",
        "base/__init__.py",
        "base/annotate.py",
        "base/ast_constants.py",
        "base/ast_utils.py",
        "base/codegen.py",
        "base/formatting.py",
        "base/fstring_utils.py",
        "base/scope.py",
        "base/test_utils.py",
        "base/token_generator.py",
    ]),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
