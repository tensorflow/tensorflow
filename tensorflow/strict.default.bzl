"""Default (OSS) build versions of Python strict rules."""

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", _py_test = "py_test")

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    _py_test(name = name, **kwargs)
